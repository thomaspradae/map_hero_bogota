#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

# ---------------- util ----------------

def clamp(x, a, b):
    return max(a, min(b, x))

def find_default_public_dir(start: Path) -> Path | None:
    """
    Try to find:
      web/bogota-plate/public
    by walking up from start.
    """
    p = start.resolve()
    for parent in [p] + list(p.parents):
        cand = parent / "web" / "bogota-plate" / "public"
        if cand.exists() and cand.is_dir():
            return cand
    return None

def load_geojson(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def get_bounds_from_aoi(fc):
    west = float("inf")
    east = float("-inf")
    south = float("inf")
    north = float("-inf")

    def visit(lon, lat):
        nonlocal west, east, south, north
        west = min(west, lon)
        east = max(east, lon)
        south = min(south, lat)
        north = max(north, lat)

    for feat in fc.get("features", []):
        g = feat.get("geometry")
        if not g:
            continue
        t = g.get("type")
        coords = g.get("coordinates", [])

        if t == "Polygon":
            for ring in coords:
                for lon, lat in ring:
                    visit(lon, lat)
        elif t == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    for lon, lat in ring:
                        visit(lon, lat)

    if not (west < east and south < north):
        raise RuntimeError("AOI bounds look invalid (west<east and south<north failed).")
    return west, south, east, north

def lonlat_to_px(lon, lat, bounds, W, H):
    west, south, east, north = bounds
    x = (lon - west) / (east - west) * W
    y = (north - lat) / (north - south) * H
    return (x, y)

def draw_polygon_with_holes(mask_draw: ImageDraw.ImageDraw, rings_px):
    """
    GeoJSON Polygon: first ring is outer, remaining rings are holes.
    We fill outer with 255, holes with 0.
    """
    if not rings_px:
        return
    outer = rings_px[0]
    if len(outer) >= 3:
        mask_draw.polygon(outer, fill=255)
    for hole in rings_px[1:]:
        if len(hole) >= 3:
            mask_draw.polygon(hole, fill=0)

def rotated_canvas_size(W, H, ang_rad):
    """
    If you rotate a WxH image about its center by ang, the smallest axis-aligned
    canvas that fits it has size:
      Wr = |W*cos| + |H*sin|
      Hr = |W*sin| + |H*cos|
    (rounded up to ints)
    """
    c = abs(math.cos(ang_rad))
    s = abs(math.sin(ang_rad))
    Wr = int(math.ceil(W * c + H * s))
    Hr = int(math.ceil(W * s + H * c))
    return Wr, Hr

# ---------------- main pipeline ----------------

def main():
    ap = argparse.ArgumentParser(description="Bake AOI mask into hillshade, then rotate + tight-crop, and emit mapping meta.")
    ap.add_argument("--aoi", type=str, default=None, help="Path to AOI GeoJSON (aoi_localidades.geojson).")
    ap.add_argument("--hillshade", type=str, default=None, help="Path to hillshade image (hillshade.png).")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory (usually .../public/bogota).")

    ap.add_argument("--pad-frac", type=float, default=0.08, help="Bounds padding fraction around AOI bbox (0..0.6).")
    ap.add_argument("--rot-deg", type=float, default=120.0, help="Rotation degrees to bake into output assets.")
    ap.add_argument("--crop-pad-px", type=int, default=8, help="Extra pixel padding around tight crop bbox (after rotation).")

    ap.add_argument("--out-hill", type=str, default="hillshade_rotcrop.webp", help="Output hillshade filename.")
    ap.add_argument("--out-mask", type=str, default="aoi_mask_rotcrop.png", help="Output AOI mask filename.")
    ap.add_argument("--out-meta", type=str, default="bounds_rotcrop.json", help="Output meta filename.")
    ap.add_argument("--webp-quality", type=int, default=90, help="WebP quality (0..100).")

    args = ap.parse_args()

    # Auto-detect public dir if not provided
    public_dir = None
    if args.outdir is None or args.aoi is None or args.hillshade is None:
        public_dir = find_default_public_dir(Path.cwd())
        if public_dir is None:
            public_dir = find_default_public_dir(Path(__file__).parent)
    if public_dir is not None:
        bogota_dir = public_dir / "bogota"
    else:
        bogota_dir = None

    aoi_path = Path(args.aoi) if args.aoi else (bogota_dir / "aoi_localidades.geojson" if bogota_dir else None)
    hill_path = Path(args.hillshade) if args.hillshade else (bogota_dir / "hillshade.png" if bogota_dir else None)
    out_dir = Path(args.outdir) if args.outdir else (bogota_dir if bogota_dir else None)

    if aoi_path is None or hill_path is None or out_dir is None:
        raise SystemExit(
            "Could not infer paths. Provide --aoi, --hillshade, and --outdir explicitly."
        )

    aoi_path = aoi_path.expanduser().resolve()
    hill_path = hill_path.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()

    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI not found: {aoi_path}")
    if not hill_path.exists():
        raise FileNotFoundError(f"Hillshade not found: {hill_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_hill = out_dir / args.out_hill
    out_mask = out_dir / args.out_mask
    out_meta = out_dir / args.out_meta

    pad_frac = clamp(float(args.pad_frac), 0.0, 0.6)
    rot_deg = float(args.rot_deg)
    rot_rad = rot_deg * math.pi / 180.0
    crop_pad = max(0, int(args.crop_pad_px))

    # --- 1) load AOI + compute padded bounds ---
    fc = load_geojson(aoi_path)
    west, south, east, north = get_bounds_from_aoi(fc)
    lonR = east - west
    latR = north - south
    bounds = (
        west - lonR * pad_frac,
        south - latR * pad_frac,
        east + lonR * pad_frac,
        north + latR * pad_frac,
    )

    # --- 2) load hillshade (RGBA) ---
    hill = Image.open(hill_path).convert("RGBA")
    W, H = hill.size

    # --- 3) build AOI alpha mask in hillshade pixel space ---
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)

    for feat in fc.get("features", []):
        g = feat.get("geometry")
        if not g:
            continue
        t = g.get("type")
        coords = g.get("coordinates", [])

        if t == "Polygon":
            rings_px = []
            for ring in coords:
                ring_px = [lonlat_to_px(lon, lat, bounds, W, H) for lon, lat in ring]
                rings_px.append(ring_px)
            draw_polygon_with_holes(draw, rings_px)

        elif t == "MultiPolygon":
            for poly in coords:
                rings_px = []
                for ring in poly:
                    ring_px = [lonlat_to_px(lon, lat, bounds, W, H) for lon, lat in ring]
                    rings_px.append(ring_px)
                draw_polygon_with_holes(draw, rings_px)

    # Apply alpha to hillshade
    hill.putalpha(mask)

    # --- 4) rotate BOTH hill+mask (expand canvas) ---
    # Mask: nearest to keep clean alpha
    mask_rot = mask.rotate(rot_deg, resample=Image.NEAREST, expand=True, fillcolor=0)
    # Hill: bicubic looks nicer
    hill_rot = hill.rotate(rot_deg, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0))

    Wr, Hr = hill_rot.size

    # --- 5) tight crop to rotated mask bbox (plus crop_pad) ---
    bbox = mask_rot.getbbox()
    if bbox is None:
        raise RuntimeError("Mask bbox is empty after rotation. Try increasing --pad-frac or check AOI geometry.")
    x0, y0, x1, y1 = bbox

    x0 = max(0, x0 - crop_pad)
    y0 = max(0, y0 - crop_pad)
    x1 = min(Wr, x1 + crop_pad)
    y1 = min(Hr, y1 + crop_pad)

    hill_out = hill_rot.crop((x0, y0, x1, y1))
    mask_out_L = mask_rot.crop((x0, y0, x1, y1))

    # Save AOI mask as RGBA with alpha = mask
    mask_out = Image.new("RGBA", mask_out_L.size, (255, 255, 255, 0))
    mask_out.putalpha(mask_out_L)

    # --- 6) emit meta for correct lon/lat mapping in the baked-rotation world ---
    # We store everything needed to map final pixels -> original pixel -> lon/lat.
    #
    # Given final pixel pf in [0..outW,0..outH]:
    #   p_rot = pf + (crop_x0, crop_y0)
    #   p_orig = R^{-1} (p_rot - c_rot) + c_orig
    #   lon/lat from p_orig using bounds and original W,H.
    #
    c_orig_x = (W - 1) * 0.5
    c_orig_y = (H - 1) * 0.5
    c_rot_x = (Wr - 1) * 0.5
    c_rot_y = (Hr - 1) * 0.5

    meta = {
        "mapping": {
            "bounds_west": bounds[0],
            "bounds_south": bounds[1],
            "bounds_east": bounds[2],
            "bounds_north": bounds[3],
            "src_W": W,
            "src_H": H,
            "rot_deg": rot_deg,
            "rot_rad": rot_rad,
            "rot_canvas_W": Wr,
            "rot_canvas_H": Hr,
            "rot_center_x": c_rot_x,
            "rot_center_y": c_rot_y,
            "src_center_x": c_orig_x,
            "src_center_y": c_orig_y,
            "crop_x0": x0,
            "crop_y0": y0,
            "out_W": hill_out.size[0],
            "out_H": hill_out.size[1],
        },
        "inputs": {
            "aoi": str(aoi_path),
            "hillshade": str(hill_path),
            "pad_frac": pad_frac,
            "crop_pad_px": crop_pad,
        },
        "outputs": {
            "hillshade": str(out_hill),
            "aoi_mask": str(out_mask),
            "meta": str(out_meta),
        },
        "notes": [
            "Final images are rotated+cropped to match on-site look.",
            "Set ROT_DEG=0 in WebGL and use meta mapping for hover lon/lat.",
        ],
    }

    # --- 7) save files ---
    # Hillshade: WebP with alpha
    hill_out.save(out_hill, "WEBP", quality=int(clamp(args.webp_quality, 0, 100)), method=6)
    # Mask: PNG
    mask_out.save(out_mask, "PNG")
    # Meta: JSON
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Wrote:")
    print(" -", out_hill)
    print(" -", out_mask)
    print(" -", out_meta)
    print("")
    print("Next step: set ROT_DEG=0 in your component, load these assets, and use meta.mapping to compute lon/lat on hover.")

if __name__ == "__main__":
    main()
