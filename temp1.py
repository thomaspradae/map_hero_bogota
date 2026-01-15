#!/usr/bin/env python3
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import rasterio

# ============================================================
# CONFIG: bake everything in here
# ============================================================

GEOTIFF_PATH   = Path("/home/t/Downloads/peaksv2blur/web/bogota-plate/public/bogota/bogota_raw_srtm.tif")
HILLSHADE_PATH = Path("/home/t/Downloads/peaksv2blur/web/bogota-plate/public/bogota/hillshade.png")
AOI_PATH       = Path("/home/t/Downloads/peaksv2blur/web/bogota-plate/public/bogota/aoi_localidades.geojson")

OUTDIR = Path("/home/t/Downloads/peaksv2blur/web/bogota-plate/public/bogota/debug_steps1_5")
OUTDIR.mkdir(parents=True, exist_ok=True)

ROT_DEG = 120.0
CROP_PAD_PX = 8

# Your test points (lon, lat)
POINTS = [
    ("Cerro Aguanoso",   -74.0531,  4.5756),
    ("Alto de la Viga",  -74.03185, 4.5719),  # adjusted
    ("Guadalupe Peak",   -74.0544,  4.5919),
    ("Monserrate Peak",  -74.0555,  4.6057),
    ("Cerro el Cable",   -74.0507,  4.6297),
]

# Cross style
CROSS_R = 8
CROSS_W = 2

# ============================================================
# GeoTIFF mapping (north-up, no rotation/shear)
# ============================================================

def lonlat_to_px_geotiff(lon, lat, lon0, lat0, dlon, dlat):
    """
    Inverse geotransform for north-up rasters:
      x = (lon - lon0)/dlon
      y = (lat - lat0)/dlat   (note dlat is usually negative)
    Returns float pixel coords (x,y) in the *source* raster grid.
    """
    x = (lon - lon0) / dlon
    y = (lat - lat0) / dlat
    return (x, y)

def px_to_lonlat_geotiff(x, y, lon0, lat0, dlon, dlat):
    lon = lon0 + x * dlon
    lat = lat0 + y * dlat
    return (lon, lat)

# ============================================================
# AOI mask helpers (minimal; no smoothing/scale to keep steps clean)
# ============================================================

import json

def load_geojson(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def draw_polygon_with_holes(mask_draw: ImageDraw.ImageDraw, rings_px):
    # GeoJSON Polygon: first ring is outer, remaining rings are holes.
    if not rings_px:
        return
    outer = rings_px[0]
    if len(outer) >= 3:
        mask_draw.polygon(outer, fill=255)
    for hole in rings_px[1:]:
        if len(hole) >= 3:
            mask_draw.polygon(hole, fill=0)

# ============================================================
# PIL rotate(expand=True) translation derivation
# ============================================================

def rotate_about_center_y_down(x, y, W, H, rot_rad):
    """
    Pure rotation about center (no expand translation).
    Uses y-down image coordinates.
    Returns coordinates in the same "unexpanded" frame.
    """
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    cosA = math.cos(rot_rad)
    sinA = math.sin(rot_rad)
    dx = x - cx
    dy = y - cy
    rx =  cosA * dx + sinA * dy
    ry = -sinA * dx + cosA * dy
    return (rx + cx, ry + cy)

def compute_pil_expand_translation_oracle(W, H, ROT_DEG, rot_rad):
    """
    Derive PIL's expand translation directly from the oracle (ground truth).
    This bypasses geometry guessing and uses PIL's actual behavior.
    """
    refs = [
        ("TL", 0, 0),
        ("TR", W-1, 0),
        ("BL", 0, H-1),
        ("BR", W-1, H-1),
        ("C",  (W-1)/2.0, (H-1)/2.0),
    ]

    txs, tys = [], []
    for name, x, y in refs:
        pil = pil_oracle_rotate_point(x, y, W, H, ROT_DEG)
        if pil is None:
            continue
        xPure, yPure = rotate_about_center_y_down(x, y, W, H, rot_rad)
        txs.append(pil[0] - xPure)
        tys.append(pil[1] - yPure)

    tx = float(np.mean(txs))
    ty = float(np.mean(tys))

    # print scatter so you see if it's constant (it should be)
    print("[ORACLE TX/TY] samples:")
    for i, (name, _, _) in enumerate(refs):
        if i < len(txs):
            print(f"  {name:3s} tx={txs[i]:9.3f}  ty={tys[i]:9.3f}")
    print(f"[ORACLE TX/TY] mean tx,ty = ({tx:.6f}, {ty:.6f})")
    print()
    return tx, ty

def apply_rotate_expand_to_point(x, y, W, H, rot_rad, rot_tx, rot_ty):
    """
    Map source pixel -> rotated(expand=True) canvas coordinates,
    using IMAGE coordinates: +x right, +y down.
    Uses oracle-derived translation for expand=True.
    """
    xPure, yPure = rotate_about_center_y_down(x, y, W, H, rot_rad)
    return (xPure + rot_tx, yPure + rot_ty)

def inverse_rotate_expand_point(xR, yR, W, H, rot_rad, rot_tx, rot_ty):
    """
    Invert apply_rotate_expand_to_point:
      1) remove expand translation
      2) inverse-rotate about center (y-down coordinates)
    """
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    cosA = math.cos(rot_rad)
    sinA = math.sin(rot_rad)

    # undo translation to get back into rotation-centered coords
    rx = xR - rot_tx
    ry = yR - rot_ty

    # inverse rotation for y-down matrix: R(-a) = R(a)^T
    dx0 = cosA * rx - sinA * ry
    dy0 = sinA * rx + cosA * ry

    x = dx0 + cx
    y = dy0 + cy
    return (x, y)

def pil_oracle_rotate_point(x, y, W, H, rot_deg):
    """
    Ground truth: draw a single white pixel at (x,y) in a blank image,
    rotate with PIL (expand=True), then find where that pixel ended up.
    Uses NEAREST so the pixel stays crisp.
    Returns (xR, yR) in rotated canvas coordinates.
    """
    # clamp to integer pixel center
    xi = int(round(x))
    yi = int(round(y))
    if not (0 <= xi < W and 0 <= yi < H):
        return None

    im = Image.new("L", (W, H), 0)
    im.putpixel((xi, yi), 255)

    rot = im.rotate(rot_deg, resample=Image.NEAREST, expand=True, fillcolor=0)
    arr = np.array(rot, dtype=np.uint8)

    ys, xs = np.where(arr == 255)
    if len(xs) == 0:
        return None

    # If rotation duplicates pixels (rare), take centroid
    xR = float(xs.mean())
    yR = float(ys.mean())
    return (xR, yR)

def debug_compare_point_mapping(points_src_xy, W, H, ROT_DEG, rot_rad, rot_tx, rot_ty):
    print("[DEBUG] PIL oracle vs our mapping")
    rot_test = Image.new("L", (W, H), 0).rotate(ROT_DEG, Image.NEAREST, expand=True)
    print("  rotated canvas size (PIL) =", rot_test.size)
    print()
    for name, xs, ys in points_src_xy:
        ours = apply_rotate_expand_to_point(xs, ys, W, H, rot_rad, rot_tx, rot_ty)
        pil  = pil_oracle_rotate_point(xs, ys, W, H, ROT_DEG)
        print(name)
        print("  src =", (xs, ys))
        print("  ours =", ours)
        print("  pil  =", pil)
        if pil is not None:
            print("  delta(pil-ours) =", (pil[0] - ours[0], pil[1] - ours[1]))
        print()

# ============================================================
# Instrumentation helpers
# ============================================================

def fmt_pt(x, y):
    return f"({x:9.3f}, {y:9.3f})"

def in_bounds(x, y, W, H, pad=0.0):
    return (-pad <= x <= (W - 1 + pad)) and (-pad <= y <= (H - 1 + pad))

def print_affine(lon0, lat0, dlon, dlat):
    # For north-up, no shear: lon = lon0 + x*dlon, lat = lat0 + y*dlat
    print("[AFFINE] North-up GeoTIFF mapping (no shear/rotation)")
    print("  lon(x,y) = lon0 + x * dlon")
    print("  lat(x,y) = lat0 + y * dlat")
    print("  inverse:")
    print("    x(lon,lat) = (lon - lon0) / dlon")
    print("    y(lon,lat) = (lat - lat0) / dlat")
    print("  numbers:")
    print(f"    lon0={lon0:.15f}  lat0={lat0:.15f}")
    print(f"    dlon={dlon:.18f}  dlat={dlat:.18f}")
    print()

def print_point_pipeline(name, lon, lat, lon0, lat0, dlon, dlat,
                         W, H, rot_rad, rot_tx, rot_ty,
                         Wr, Hr, crop_x0=None, crop_y0=None, crop_x1=None, crop_y1=None):
    xs, ys = lonlat_to_px_geotiff(lon, lat, lon0, lat0, dlon, dlat)
    lon2, lat2 = px_to_lonlat_geotiff(xs, ys, lon0, lat0, dlon, dlat)

    xr, yr = apply_rotate_expand_to_point(xs, ys, W, H, rot_rad, rot_tx, rot_ty)

    print(f"{name}")
    print(f"  lon,lat           = ({lon:.6f}, {lat:.6f})")
    print(f"  src px (float)    = {fmt_pt(xs, ys)}  in_bounds={in_bounds(xs, ys, W, H, pad=1.0)}")
    print(f"  roundtrip lon/lat = ({lon2:.6f}, {lat2:.6f})  err=({lon2-lon:+.3e}, {lat2-lat:+.3e})")
    print(f"  rot px (expand)   = {fmt_pt(xr, yr)}  in_bounds={in_bounds(xr, yr, Wr, Hr, pad=1.0)}")

    if crop_x0 is not None:
        xc = xr - crop_x0
        yc = yr - crop_y0
        in_crop = (crop_x0 <= xr <= crop_x1) and (crop_y0 <= yr <= crop_y1)
        print(f"  crop bbox rot     = ({crop_x0},{crop_y0})..({crop_x1},{crop_y1})  point_in_bbox={in_crop}")
        print(f"  cropped px        = {fmt_pt(xc, yc)}  in_bounds={in_bounds(xc, yc, crop_x1-crop_x0, crop_y1-crop_y0, pad=1.0)}")

    # rotation invert sanity
    xs2, ys2 = inverse_rotate_expand_point(xr, yr, W, H, rot_rad, rot_tx, rot_ty)
    print(f"  inv-rot back src  = {fmt_pt(xs2, ys2)}  err_src=({xs2-xs:+.3e}, {ys2-ys:+.3e})")
    print()

# ============================================================
# Drawing helpers
# ============================================================

def draw_cross(draw: ImageDraw.ImageDraw, x, y, r=CROSS_R, w=CROSS_W, color=(255,0,0)):
    x = int(round(x))
    y = int(round(y))
    draw.line([(x - r, y), (x + r, y)], fill=color, width=w)
    draw.line([(x, y - r), (x, y + r)], fill=color, width=w)

def draw_label(draw: ImageDraw.ImageDraw, x, y, text, color=(255,0,0)):
    # simple label offset
    x = int(round(x)) + 10
    y = int(round(y)) - 10
    draw.text((x, y), text, fill=color)

# ============================================================
# MAIN: Steps 1 through 5 + visual tests
# ============================================================

def main():
    # ----------------------------
    # Step 1) Read GeoTIFF geotransform
    # ----------------------------
    if not GEOTIFF_PATH.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {GEOTIFF_PATH}")
    with rasterio.open(GEOTIFF_PATH) as ds:
        T = ds.transform
        lon0 = float(T.c)
        lat0 = float(T.f)
        dlon = float(T.a)
        dlat = float(T.e)
        geotiff_W = ds.width
        geotiff_H = ds.height

    print("[STEP 1] GeoTIFF transform")
    print_affine(lon0, lat0, dlon, dlat)
    print("  size       =", geotiff_W, geotiff_H)
    print()

    # ----------------------------
    # Step 2) Load hillshade (must be 1:1 grid)
    # ----------------------------
    if not HILLSHADE_PATH.exists():
        raise FileNotFoundError(f"Hillshade not found: {HILLSHADE_PATH}")
    hill = Image.open(HILLSHADE_PATH).convert("RGBA")
    W, H = hill.size
    if (W, H) != (geotiff_W, geotiff_H):
        raise RuntimeError(
            f"Hillshade size {W}x{H} != GeoTIFF size {geotiff_W}x{geotiff_H}.\n"
            "They must be pixel-identical for lon/lat→px to match."
        )
    print("[STEP 2] Hillshade loaded and matches GeoTIFF grid.")
    print()

    # ----------------------------
    # TEST 01: draw points on original hillshade (before any AOI/mask/rotate)
    # ----------------------------
    t01 = hill.copy()
    d01 = ImageDraw.Draw(t01)
    print("[TEST 01] Points on original hillshade:")
    for name, lon, lat in POINTS:
        x, y = lonlat_to_px_geotiff(lon, lat, lon0, lat0, dlon, dlat)
        print(f"  {name:16s}  lon,lat=({lon:.6f},{lat:.6f})  ->  x,y=({x:.2f},{y:.2f})")
        draw_cross(d01, x, y)
        draw_label(d01, x, y, name)
    t01_path = OUTDIR / "test01_points_on_hillshade.png"
    t01.save(t01_path)
    print("  wrote:", t01_path)
    print()

    # ----------------------------
    # Step 3) Build AOI mask in original pixel space (no smoothing/scale; keep it simple)
    # ----------------------------
    if not AOI_PATH.exists():
        raise FileNotFoundError(f"AOI GeoJSON not found: {AOI_PATH}")
    fc = load_geojson(AOI_PATH)
    mask = Image.new("L", (W, H), 0)
    md = ImageDraw.Draw(mask)

    feat_count = 0
    for feat in fc.get("features", []):
        g = feat.get("geometry")
        if not g:
            continue
        feat_count += 1
        t = g.get("type")
        coords = g.get("coordinates", [])

        if t == "Polygon":
            rings_px = []
            for ring in coords:
                ring_px = [lonlat_to_px_geotiff(lon, lat, lon0, lat0, dlon, dlat) for lon, lat in ring]
                rings_px.append(ring_px)
            draw_polygon_with_holes(md, rings_px)

        elif t == "MultiPolygon":
            for poly in coords:
                rings_px = []
                for ring in poly:
                    ring_px = [lonlat_to_px_geotiff(lon, lat, lon0, lat0, dlon, dlat) for lon, lat in ring]
                    rings_px.append(ring_px)
                draw_polygon_with_holes(md, rings_px)

    print(f"[STEP 3] AOI mask built from {feat_count} features.")
    print()

    # AOI mask bbox sanity
    bbox_src = mask.getbbox()
    if bbox_src is None:
        raise RuntimeError("AOI mask bbox empty in SOURCE space. The AOI did not draw at all.")
    x0s, y0s, x1s, y1s = bbox_src
    print("[STEP 3 DEBUG] AOI bbox in SOURCE pixels")
    print("  bbox_src =", bbox_src)
    print("  width,height =", (x1s-x0s), (y1s-y0s))
    print("  in_bounds bbox corners:",
          in_bounds(x0s, y0s, W, H, pad=1.0),
          in_bounds(x1s-1, y1s-1, W, H, pad=1.0))
    print()

    # Save a debug image: mask + bbox
    dbg_mask = Image.new("RGBA", (W, H), (0,0,0,0))
    dbg_mask.putalpha(mask)
    dmb = ImageDraw.Draw(dbg_mask)
    dmb.rectangle([x0s, y0s, x1s, y1s], outline=(0,255,0,255), width=3)
    dbg_mask_path = OUTDIR / "step03_mask_with_bbox.png"
    dbg_mask.save(dbg_mask_path)
    print("  wrote:", dbg_mask_path)
    print()

    # TEST 02: overlay mask on hillshade so you can sanity check AOI position
    t02 = hill.copy()
    t02.putalpha(mask)  # just for visualization: show AOI only
    # add points too
    d02 = ImageDraw.Draw(t02)
    for name, lon, lat in POINTS:
        x, y = lonlat_to_px_geotiff(lon, lat, lon0, lat0, dlon, dlat)
        draw_cross(d02, x, y)
        draw_label(d02, x, y, name)
    t02_path = OUTDIR / "test02_mask_on_hillshade.png"
    t02.save(t02_path)
    print("[TEST 02] Masked hillshade (AOI alpha) + points wrote:", t02_path)
    print()

    # ----------------------------
    # Step 4) Rotate BOTH (expand=True)
    # ----------------------------
    rot_rad = ROT_DEG * math.pi / 180.0

    mask_rot = mask.rotate(ROT_DEG, resample=Image.NEAREST, expand=True, fillcolor=0)
    hill_rot = hill.rotate(ROT_DEG, resample=Image.BICUBIC, expand=True, fillcolor=(0,0,0,0))
    Wr, Hr = hill_rot.size

    # Use oracle to derive PIL's actual expand translation
    rot_tx, rot_ty = compute_pil_expand_translation_oracle(W, H, ROT_DEG, rot_rad)

    print("[STEP 4] Rotated(expand=True)")
    print("  ROT_DEG =", ROT_DEG)
    print("  rotated canvas size =", Wr, Hr)
    print("  computed rot_tx, rot_ty =", rot_tx, rot_ty)
    print()

    # Rotated mask bbox sanity
    bbox_rot = mask_rot.getbbox()
    if bbox_rot is None:
        raise RuntimeError("Mask bbox empty after rotation. AOI disappeared post-rotate.")
    print("[STEP 4 DEBUG] Rotated mask bbox (PIL computed)")
    print("  bbox_rot =", bbox_rot)
    print()

    # Save rotated mask with bbox drawn
    dbg_rot_mask = Image.new("RGBA", (Wr, Hr), (0,0,0,0))
    dbg_rot_mask.putalpha(mask_rot)
    drm = ImageDraw.Draw(dbg_rot_mask)
    x0r, y0r, x1r, y1r = bbox_rot
    drm.rectangle([x0r, y0r, x1r, y1r], outline=(0,255,0,255), width=3)
    dbg_rot_mask_path = OUTDIR / "step04_rotated_mask_with_bbox.png"
    dbg_rot_mask.save(dbg_rot_mask_path)
    print("  wrote:", dbg_rot_mask_path)
    print()

    # Optional: validate the translation by rotating SOURCE corners and comparing with PIL size
    corners_src = [(0,0), (W-1,0), (W-1,H-1), (0,H-1)]
    corners_exp = []
    for (x,y) in corners_src:
        xr, yr = apply_rotate_expand_to_point(x, y, W, H, rot_rad, rot_tx, rot_ty)
        corners_exp.append((xr, yr))
    minx = min(p[0] for p in corners_exp)
    miny = min(p[1] for p in corners_exp)
    maxx = max(p[0] for p in corners_exp)
    maxy = max(p[1] for p in corners_exp)
    print("[STEP 4 DEBUG] Rotated(expand) SOURCE corner envelope (via our math)")
    print(f"  min=({minx:.3f},{miny:.3f}) max=({maxx:.3f},{maxy:.3f})")
    print(f"  predicted size ~ ({(maxx-minx):.3f}, {(maxy-miny):.3f})  actual PIL size=({Wr},{Hr})")
    print()

    print("[POINT PIPELINE] After Step 4 (src -> rot)")
    for name, lon, lat in POINTS:
        print_point_pipeline(
            name, lon, lat,
            lon0, lat0, dlon, dlat,
            W, H, rot_rad, rot_tx, rot_ty,
            Wr, Hr
        )

    # PIL oracle comparison
    points_src_xy = []
    for name, lon, lat in POINTS:
        xs, ys = lonlat_to_px_geotiff(lon, lat, lon0, lat0, dlon, dlat)
        points_src_xy.append((name, xs, ys))

    debug_compare_point_mapping(points_src_xy, W, H, ROT_DEG, rot_rad, rot_tx, rot_ty)

    # TEST 03: draw points on rotated image - test both +angle and -angle
    # PIL rotates CCW in image coords (+y down), which flips the sense vs math coords
    t03 = hill_rot.copy()
    d03 = ImageDraw.Draw(t03)

    # Get rotated AOI bbox for validation
    x0r, y0r, x1r, y1r = bbox_rot

    # Compute transforms for both angle signs (using oracle for ground truth)
    rot_rad_pos = ROT_DEG * math.pi / 180.0
    rot_rad_neg = -ROT_DEG * math.pi / 180.0

    rot_tx_pos, rot_ty_pos = compute_pil_expand_translation_oracle(W, H, ROT_DEG, rot_rad_pos)
    rot_tx_neg, rot_ty_neg = compute_pil_expand_translation_oracle(W, H, -ROT_DEG, rot_rad_neg)

    print("[TEST 03] Testing +angle vs -angle rotation (PIL convention check)")
    print("  rot_tx_pos, rot_ty_pos =", rot_tx_pos, rot_ty_pos)
    print("  rot_tx_neg, rot_ty_neg =", rot_tx_neg, rot_ty_neg)
    print("  rotated AOI bbox =", bbox_rot)
    print()

    for name, lon, lat in POINTS:
        xs, ys = lonlat_to_px_geotiff(lon, lat, lon0, lat0, dlon, dlat)

        # current (likely wrong - standard math convention)
        xr1, yr1 = apply_rotate_expand_to_point(xs, ys, W, H, rot_rad_pos, rot_tx_pos, rot_ty_pos)
        in_bbox_pos = (x0r <= xr1 <= x1r and y0r <= yr1 <= y1r)
        
        # flipped angle (likely correct - PIL image coordinate convention)
        xr2, yr2 = apply_rotate_expand_to_point(xs, ys, W, H, rot_rad_neg, rot_tx_neg, rot_ty_neg)
        in_bbox_neg = (x0r <= xr2 <= x1r and y0r <= yr2 <= y1r)

        print(f"  {name:20s}  in_bbox(+ang)={str(in_bbox_pos):5s}  in_bbox(-ang)={str(in_bbox_neg):5s}")

        # red: +angle (standard math)
        draw_cross(d03, xr1, yr1, color=(255,0,0))
        draw_label(d03, xr1, yr1, name + " +ang", color=(255,0,0))

        # cyan: -angle (PIL image coords)
        draw_cross(d03, xr2, yr2, color=(0,255,255))
        draw_label(d03, xr2, yr2, name + " -ang", color=(0,255,255))

    print()

    t03_path = OUTDIR / "test03_points_on_rotated.png"
    t03.save(t03_path)
    print("[TEST 03] Rotated image + transformed points (red=+ang, cyan=-ang) wrote:", t03_path)
    print()

    # ----------------------------
    # Step 5) Tight crop using rotated mask bbox (+pad)
    # ----------------------------
    bbox = mask_rot.getbbox()
    if bbox is None:
        raise RuntimeError("Mask bbox empty after rotation (AOI disappeared).")
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - CROP_PAD_PX)
    y0 = max(0, y0 - CROP_PAD_PX)
    x1 = min(Wr, x1 + CROP_PAD_PX)
    y1 = min(Hr, y1 + CROP_PAD_PX)

    hill_crop = hill_rot.crop((x0, y0, x1, y1)).convert("RGBA")
    mask_crop = mask_rot.crop((x0, y0, x1, y1))

    print("[STEP 5] Cropped")
    print("  crop bbox on rotated canvas =", (x0, y0, x1, y1))
    print("  cropped size =", hill_crop.size)
    print()

    print("[POINT PIPELINE] After Step 5 (src -> rot -> crop)")
    for name, lon, lat in POINTS:
        print_point_pipeline(
            name, lon, lat,
            lon0, lat0, dlon, dlat,
            W, H, rot_rad, rot_tx, rot_ty,
            Wr, Hr,
            crop_x0=x0, crop_y0=y0, crop_x1=x1, crop_y1=y1
        )

    # TEST 04: show crop bbox on rotated image
    t04 = hill_rot.copy()
    d04 = ImageDraw.Draw(t04)
    d04.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)
    t04_path = OUTDIR / "test04_crop_box_on_rotated.png"
    t04.save(t04_path)
    print("[TEST 04] Rotated image with crop box wrote:", t04_path)
    print()

    # TEST 05: draw points on cropped output
    t05 = hill_crop.copy()
    d05 = ImageDraw.Draw(t05)
    for name, lon, lat in POINTS:
        xs, ys = lonlat_to_px_geotiff(lon, lat, lon0, lat0, dlon, dlat)
        xr, yr = apply_rotate_expand_to_point(xs, ys, W, H, rot_rad, rot_tx, rot_ty)
        # crop shift
        xc = xr - x0
        yc = yr - y0
        draw_cross(d05, xc, yc)
        draw_label(d05, xc, yc, name)

    t05_path = OUTDIR / "test05_points_on_cropped.png"
    t05.save(t05_path)
    print("[TEST 05] Cropped image + points wrote:", t05_path)
    print()

    # Also save the cropped mask so you can inspect it if needed
    mask_path = OUTDIR / "step05_mask_cropped.png"
    mask_crop.save(mask_path)
    print("Saved cropped mask:", mask_path)
    print()
    print("Done. Open the PNGs in:", OUTDIR)

if __name__ == "__main__":
    main()
