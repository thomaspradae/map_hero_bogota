#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.warp import transform as rio_transform
except ImportError as e:
    raise SystemExit(
        "Missing rasterio. Install it in your venv:\n"
        "  pip install rasterio\n"
        f"Original error: {e}"
    )


# ---------------- util ----------------

def clamp(x, a, b):
    return max(a, min(b, x))

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def bilinear_sample(arr2d: np.ndarray, x: np.ndarray, y: np.ndarray, nodata: float):
    """
    Bilinear sample of arr2d at floating pixel coords x,y.
    - arr2d is shape (H,W)
    - x,y are same shape (N,) or (H,W) floats in pixel coordinates
    Returns float32 with nodata where any neighbor is nodata or out of bounds.
    """
    H, W = arr2d.shape

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # mask of valid neighbors
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < W) & (y1 < H)

    out = np.full_like(x, fill_value=nodata, dtype=np.float32)
    if not np.any(valid):
        return out

    xv = x[valid]
    yv = y[valid]
    x0v = x0[valid]; y0v = y0[valid]
    x1v = x1[valid]; y1v = y1[valid]

    Ia = arr2d[y0v, x0v].astype(np.float32)
    Ib = arr2d[y0v, x1v].astype(np.float32)
    Ic = arr2d[y1v, x0v].astype(np.float32)
    Id = arr2d[y1v, x1v].astype(np.float32)

    # If any neighbor is nodata, treat as nodata (keeps things honest)
    good = (Ia != nodata) & (Ib != nodata) & (Ic != nodata) & (Id != nodata)
    if not np.any(good):
        return out

    xv2 = xv[good]
    yv2 = yv[good]
    x0g = x0v[good]; y0g = y0v[good]
    x1g = x1v[good]; y1g = y1v[good]

    Ia = arr2d[y0g, x0g].astype(np.float32)
    Ib = arr2d[y0g, x1g].astype(np.float32)
    Ic = arr2d[y1g, x0g].astype(np.float32)
    Id = arr2d[y1g, x1g].astype(np.float32)

    dx = (xv2 - x0g.astype(np.float32))
    dy = (yv2 - y0g.astype(np.float32))

    wa = (1 - dx) * (1 - dy)
    wb = dx * (1 - dy)
    wc = (1 - dx) * dy
    wd = dx * dy

    samp = wa * Ia + wb * Ib + wc * Ic + wd * Id

    out_valid = out[valid]
    out_valid_good = out_valid.copy()
    out_valid_good[good] = samp
    out[valid] = out_valid_good

    return out

def unpack_mapping(bounds_rotcrop_json):
    """
    Accept either:
      - the full meta object written by bake.py
      - or a dict that already has 'mapping'
    """
    if "mapping" in bounds_rotcrop_json:
        m = bounds_rotcrop_json["mapping"]
    else:
        m = bounds_rotcrop_json

    # required keys:
    req = [
        "bounds_west","bounds_south","bounds_east","bounds_north",
        "src_W","src_H",
        "rot_deg","rot_rad",
        "rot_canvas_W","rot_canvas_H",
        "rot_center_x","rot_center_y",
        "src_center_x","src_center_y",
        "crop_x0","crop_y0",
        "out_W","out_H",
        "squished"
    ]
    for k in req:
        if k not in m:
            raise KeyError(f"Missing mapping key '{k}' in bounds_rotcrop.json")

    return m

def outpx_to_lonlat(m, out_x, out_y):
    """
    Inverse mapping using PIL's actual expand translation:
      out -> (undo squish) -> undo crop -> undo PIL translation -> inverse rotate -> src px -> lon/lat
    """
    # 1) Undo squish
    if bool(m["squished"]):
        preW = float(m["pre_squish_W"])
        preH = float(m["pre_squish_H"])
        out_W = float(m["out_W"])
        out_H = float(m["out_H"])
        sx = preW / out_W
        sy = preH / out_H
        crop_x = out_x * sx
        crop_y = out_y * sy
    else:
        crop_x = out_x.copy()
        crop_y = out_y.copy()

    # 2) Undo crop
    rot_x = crop_x + float(m["crop_x0"])
    rot_y = crop_y + float(m["crop_y0"])

    # 3) Undo PIL's expand translation
    rot_tx = float(m["rot_tx"])
    rot_ty = float(m["rot_ty"])
    
    rx = rot_x - rot_tx
    ry = rot_y - rot_ty

    # 4) Inverse rotate around src center
    ang = float(m["rot_rad"])
    cosA = math.cos(ang)
    sinA = math.sin(ang)
    
    src_W = float(m["src_W"])
    src_H = float(m["src_H"])
    src_cx = src_W / 2.0
    src_cy = src_H / 2.0
    
    # Inverse rotation (rotate by -ang)
    src_x = (cosA * rx + sinA * ry) + src_cx
    src_y = (-sinA * rx + cosA * ry) + src_cy

    # 5) src pixel -> lon/lat
    west = float(m["bounds_west"])
    south = float(m["bounds_south"])
    east = float(m["bounds_east"])
    north = float(m["bounds_north"])

    lon = west + (src_x / src_W) * (east - west)
    lat = north - (src_y / src_H) * (north - south)
    
    return lon.astype(np.float64), lat.astype(np.float64)


def main():
    ap = argparse.ArgumentParser(
        description="Bake elevation into RG PNG aligned to hillshade_paper_rotcrop.webp pixel space."
    )
    ap.add_argument("--meta", required=True, help="Path to bounds_rotcrop.json produced by bake.py")
    ap.add_argument("--aoi-mask", required=True, help="Path to aoi_mask_rotcrop.png (alpha mask, 910x820)")
    ap.add_argument("--dem", required=True, help="Path to DEM GeoTIFF (e.g. bogota_raw_srtm.tif)")
    ap.add_argument("--outdir", required=True, help="Output directory (same public/bogota)")

    ap.add_argument("--out-elev", default="elev_rg_rotcrop.png", help="Output elevation RG PNG")
    ap.add_argument("--out-meta", default="elev_meta.json", help="Output elevation meta json")

    ap.add_argument("--clip-to-aoi", action="store_true", help="Multiply alpha by AOI alpha (recommended).")
    ap.add_argument("--nodata-fill", type=float, default=-9999.0, help="Fallback nodata value if DEM lacks one.")

    args = ap.parse_args()

    meta_path = Path(args.meta).expanduser().resolve()
    mask_path = Path(args.aoi_mask).expanduser().resolve()
    dem_path = Path(args.dem).expanduser().resolve()
    out_dir = Path(args.outdir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not meta_path.exists():
        raise FileNotFoundError(f"Meta not found: {meta_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"AOI mask not found: {mask_path}")
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    meta = load_json(meta_path)
    m = unpack_mapping(meta)

    # Load AOI alpha (must match output size)
    mask_rgba = Image.open(mask_path).convert("RGBA")
    outW, outH = mask_rgba.size
    if int(m["out_W"]) != outW or int(m["out_H"]) != outH:
        raise RuntimeError(
            f"Size mismatch: meta out_W/out_H={m['out_W']}x{m['out_H']} but aoi_mask is {outW}x{outH}"
        )
    aoi_alpha = np.array(mask_rgba, dtype=np.uint8)[..., 3]  # 0..255

    # Open DEM and read band 1 into memory
    with rasterio.open(dem_path) as ds:
        dem = ds.read(1).astype(np.float32)
        dem_nodata = ds.nodata
        if dem_nodata is None:
            dem_nodata = float(args.nodata_fill)

        dem_crs = ds.crs
        dem_transform = ds.transform
        dem_W = ds.width
        dem_H = ds.height

        # Build output pixel grid
        xs = np.arange(outW, dtype=np.float32)
        ys = np.arange(outH, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)  # shape (outH,outW)

        # Convert output px -> lon/lat
        lon, lat = outpx_to_lonlat(m, grid_x, grid_y)

        # Transform lon/lat (EPSG:4326) into DEM CRS if needed
        if dem_crs is None:
            raise RuntimeError("DEM CRS is None; need a CRS to transform lon/lat into DEM coords.")

        # rasterio expects x,y in DEM CRS (usually meters) to compute pixel coords
        if str(dem_crs).lower().find("4326") != -1:
            X = lon
            Y = lat
        else:
            # transform expects arrays in sequence
            X, Y = rio_transform("EPSG:4326", dem_crs, lon.ravel().tolist(), lat.ravel().tolist())
            X = np.array(X, dtype=np.float64).reshape(outH, outW)
            Y = np.array(Y, dtype=np.float64).reshape(outH, outW)

        # Convert DEM-world coords -> DEM pixel coords
        # For affine transform: col,row = ~transform * (x,y)
        invT = ~dem_transform
        dem_col = invT.a * X + invT.b * Y + invT.c
        dem_row = invT.d * X + invT.e * Y + invT.f

        dem_col = dem_col.astype(np.float32)
        dem_row = dem_row.astype(np.float32)

        # Sample DEM bilinearly
        elev = bilinear_sample(dem, dem_col, dem_row, nodata=dem_nodata)  # shape (outH,outW)

    # Mask out outside AOI if requested
    alpha = aoi_alpha.copy()
    if args.clip_to_aoi:
        # keep alpha as AOI alpha, and zero elevation outside AOI for sanity
        outside = (aoi_alpha == 0)
        elev = elev.copy()
        elev[outside] = np.nan

    # Compute elev range from valid pixels
    valid = np.isfinite(elev)
    if not np.any(valid):
        raise RuntimeError("No valid elevation samples produced. Check mapping/DEM CRS/bounds.")

    elev_min = float(np.nanmin(elev))
    elev_max = float(np.nanmax(elev))

    # Normalize to 0..65535 and pack into RG
    # Note: we clamp because bilinear + nodata edges can create small weird values.
    norm = (elev - elev_min) / max(1e-9, (elev_max - elev_min))
    norm = np.clip(norm, 0.0, 1.0)
    q = (norm * 65535.0 + 0.5).astype(np.uint16)

    hi = ((q >> 8) & 0xFF).astype(np.uint8)
    lo = (q & 0xFF).astype(np.uint8)

    out = np.zeros((outH, outW, 4), dtype=np.uint8)
    out[..., 0] = hi
    out[..., 1] = lo
    out[..., 2] = 0
    out[..., 3] = alpha

    out_img = Image.fromarray(out, mode="RGBA")

    out_elev_path = out_dir / args.out_elev
    out_meta_path = out_dir / args.out_meta

    out_img.save(out_elev_path, "PNG")

    elev_meta = {
        "elev_min_m": elev_min,
        "elev_max_m": elev_max,
        "encoding": "RG16",
        "decode": {
            "value_u16": "v = R*256 + G",
            "elev_m": "elev = elev_min_m + (v/65535)*(elev_max_m - elev_min_m)"
        },
        "stats": {
            "valid_px": int(np.sum(valid)),
            "total_px": int(outW * outH),
            "valid_frac": float(np.sum(valid) / float(outW * outH)),
        },
        "inputs": {
            "meta": str(meta_path),
            "aoi_mask": str(mask_path),
            "dem": str(dem_path),
        },
        "notes": [
            "This image is aligned 1:1 with hillshade_paper_rotcrop.webp pixel space.",
            "Alpha channel is AOI alpha; outside AOI alpha is 0.",
        ],
        "WARNING": [
            "This script requires bounds_rotcrop.json to contain pre_squish_W/pre_squish_H if squished=true.",
            "If missing, either add those fields in bake.py OR re-run bake.py with squish disabled.",
        ],
    }

    write_json(out_meta_path, elev_meta)

    print("Wrote:")
    print(" -", out_elev_path)
    print(" -", out_meta_path)
    print("Elevation range (m):", elev_min, "to", elev_max)


if __name__ == "__main__":
    main()
