#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ---------------- util ----------------

def clamp(x, a, b):
    return max(a, min(b, x))

def load_geojson(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def iter_lonlat_from_geojson(fc):
    """
    GeoJSON coordinates are (lon, lat). Ignore any axis-order WKT weirdness.
    Yields (lon, lat) for every vertex in all rings.
    """
    for feat in fc.get("features", []):
        g = feat.get("geometry")
        if not g:
            continue
        t = g.get("type")
        coords = g.get("coordinates", [])
        if t == "Polygon":
            for ring in coords:
                for lon, lat in ring:
                    yield float(lon), float(lat)
        elif t == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    for lon, lat in ring:
                        yield float(lon), float(lat)

def geotiff_lonlat_to_px(lon, lat, lon0, lat0, dlon, dlat):
    """
    GeoTIFF inverse geotransform for north-up rasters (no rotation/shear).
    lon0,lat0 = origin at pixel (0,0) upper-left.
    dlon > 0, dlat < 0 typically.
    Returns float pixel coords (x,y).
    """
    x = (lon - lon0) / dlon
    y = (lat - lat0) / dlat
    return (x, y)

def laplacian_smooth_ring(pts, passes, alpha=0.5):
    if passes <= 0 or len(pts) < 4:
        return pts
    curr = np.array(pts, dtype=np.float32)
    n = len(curr)
    for _ in range(passes):
        nxt = curr.copy()
        for i in range(n - 1):  # ignore last, keep ring closed
            prev_i = (i - 1) if i > 0 else (n - 2)
            next_i = (i + 1) if i < (n - 2) else 0
            avg = 0.5 * (curr[prev_i] + curr[next_i])
            nxt[i] = (1 - alpha) * curr[i] + alpha * avg
        nxt[n - 1] = nxt[0]
        curr = nxt
    return [tuple(p) for p in curr]

def scale_ring_about_centroid(pts, s):
    if abs(s - 1.0) < 1e-6 or len(pts) < 3:
        return pts
    arr = np.array(pts, dtype=np.float32)
    cx, cy = arr[:, 0].mean(), arr[:, 1].mean()
    arr[:, 0] = cx + (arr[:, 0] - cx) * s
    arr[:, 1] = cy + (arr[:, 1] - cy) * s
    return [tuple(p) for p in arr]

def draw_polygon_with_holes(mask_draw: ImageDraw.ImageDraw, rings_px):
    if not rings_px:
        return
    outer = rings_px[0]
    if len(outer) >= 3:
        mask_draw.polygon(outer, fill=255)
    for hole in rings_px[1:]:
        if len(hole) >= 3:
            mask_draw.polygon(hole, fill=0)

# ---------------- "paper look" bake ----------------
def apply_paper_look_rgba(img_rgba: Image.Image, hill_alpha: float, hill_gamma: float, paper_rgb, exposure: float):
    hill_alpha = float(clamp(hill_alpha, 0.0, 1.0))
    hill_gamma = max(1e-6, float(hill_gamma))
    exposure = float(exposure)
    paper = np.array(paper_rgb, dtype=np.float32)

    arr = np.array(img_rgba, dtype=np.uint8)
    rgb = arr[..., :3].astype(np.float32) / 255.0
    a = arr[..., 3:4].copy()

    hl = (rgb[..., 0] + rgb[..., 1] + rgb[..., 2]) / 3.0
    hl = np.clip(hl, 0.0, 1.0)
    hl = np.power(hl, hill_gamma)

    base = (1.0 - hill_alpha) * paper[None, None, :] + hill_alpha * hl[..., None]
    base = np.clip(base * exposure, 0.0, 1.0)

    out = np.concatenate([(base * 255.0 + 0.5).astype(np.uint8), a], axis=-1)
    return Image.fromarray(out, mode="RGBA")

# ---------------- main pipeline ----------------

def main():
    ap = argparse.ArgumentParser(
        description="AOI alpha -> rotate -> tight-crop -> bake paper look -> optional squish. AOI->pixel uses GeoTIFF geotransform."
    )
    ap.add_argument("--aoi", required=True, help="Path to AOI GeoJSON.")
    ap.add_argument("--hillshade", required=True, help="Path to hillshade image (must match GeoTIFF pixel grid).")
    ap.add_argument("--outdir", required=True, help="Output directory.")

    # GeoTIFF geotransform (north-up)
    ap.add_argument("--lon0", type=float, required=True, help="GeoTIFF origin lon at pixel (0,0).")
    ap.add_argument("--lat0", type=float, required=True, help="GeoTIFF origin lat at pixel (0,0).")
    ap.add_argument("--dlon", type=float, required=True, help="GeoTIFF pixel size in lon (degrees/pixel).")
    ap.add_argument("--dlat", type=float, required=True, help="GeoTIFF pixel size in lat (degrees/pixel, usually negative).")

    # rotation/crop
    ap.add_argument("--rot-deg", type=float, default=120.0, help="Rotation degrees to bake into output assets.")
    ap.add_argument("--crop-pad-px", type=int, default=8, help="Extra pixel padding around tight crop bbox (post-rotation).")
    ap.add_argument("--crop-pad-frac", type=float, default=0.0, help="Extra bbox padding as fraction of bbox size (post-rotation).")

    # paper look
    ap.add_argument("--hill-alpha", type=float, default=0.30)
    ap.add_argument("--hill-gamma", type=float, default=0.85)
    ap.add_argument("--paper", type=str, default="0.975,0.972,0.965")
    ap.add_argument("--exposure", type=float, default=1.0)

    # AOI conditioning
    ap.add_argument("--clip-smooth-passes", type=int, default=0)
    ap.add_argument("--clip-scale", type=float, default=1.0)

    # resize
    ap.add_argument("--squish", action="store_true")
    ap.add_argument("--target-w", type=int, default=910)
    ap.add_argument("--target-h", type=int, default=820)

    # outputs
    ap.add_argument("--out-hill", type=str, default="hillshade_paper_rotcrop.webp")
    ap.add_argument("--out-mask", type=str, default="aoi_mask_rotcrop.png")
    ap.add_argument("--out-meta", type=str, default="mapping_rotcrop.json")
    ap.add_argument("--webp-quality", type=int, default=92)

    args = ap.parse_args()

    aoi_path = Path(args.aoi).expanduser().resolve()
    hill_path = Path(args.hillshade).expanduser().resolve()
    out_dir = Path(args.outdir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fc = load_geojson(aoi_path)

    paper = tuple(float(x) for x in args.paper.split(","))
    if len(paper) != 3:
        raise ValueError("--paper must be 'r,g,b' in 0..1")

    # --- 1) load hillshade (RGBA) ---
    hill = Image.open(hill_path).convert("RGBA")
    W, H = hill.size

    # --- 2) build AOI alpha mask in *GeoTIFF pixel space* (same as hillshade pixel grid) ---
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)

    lon0, lat0, dlon, dlat = args.lon0, args.lat0, args.dlon, args.dlat

    for feat in fc.get("features", []):
        g = feat.get("geometry")
        if not g:
            continue
        t = g.get("type")
        coords = g.get("coordinates", [])

        def ring_to_px(ring):
            ring_px = [geotiff_lonlat_to_px(lon, lat, lon0, lat0, dlon, dlat) for lon, lat in ring]
            # ensure closed
            if len(ring_px) >= 3 and (abs(ring_px[0][0] - ring_px[-1][0]) > 1e-3 or abs(ring_px[0][1] - ring_px[-1][1]) > 1e-3):
                ring_px = ring_px + [ring_px[0]]

            sp = max(0, int(args.clip_smooth_passes))
            if sp > 0:
                ring_px = laplacian_smooth_ring(ring_px, sp, alpha=0.5)

            cs = float(args.clip_scale)
            if abs(cs - 1.0) > 1e-6:
                ring_px = scale_ring_about_centroid(ring_px, cs)

            return ring_px

        if t == "Polygon":
            rings_px = [ring_to_px(ring) for ring in coords]
            draw_polygon_with_holes(draw, rings_px)

        elif t == "MultiPolygon":
            for poly in coords:
                rings_px = [ring_to_px(ring) for ring in poly]
                draw_polygon_with_holes(draw, rings_px)

    # Apply alpha (this is geometry-neutral; doesn’t change pixel coords)
    hill.putalpha(mask)

    # --- 3) rotate BOTH (expand canvas) ---
    rot_deg = float(args.rot_deg)
    rot_rad = rot_deg * math.pi / 180.0
    cosA = math.cos(rot_rad)
    sinA = math.sin(rot_rad)

    mask_rot = mask.rotate(rot_deg, resample=Image.NEAREST, expand=True, fillcolor=0)
    hill_rot = hill.rotate(rot_deg, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0))
    Wr, Hr = hill_rot.size

    # Compute translation implied by expand=True (this is the missing piece most people lose)
    src_cx = (W - 1) * 0.5
    src_cy = (H - 1) * 0.5

    corners = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float64)
    corners_rot = []

    for x, y in corners:
        dx = x - src_cx
        dy = y - src_cy
        rx = cosA * dx - sinA * dy + src_cx
        ry = sinA * dx + cosA * dy + src_cy
        corners_rot.append((rx, ry))

    corners_rot = np.array(corners_rot)
    min_x = float(corners_rot[:, 0].min())
    min_y = float(corners_rot[:, 1].min())

    # PIL shifts by (-min_x, -min_y) so the rotated bbox starts at (0,0)
    rot_tx = -min_x
    rot_ty = -min_y

    print(f"[DEBUG] rot canvas: {Wr}x{Hr}")
    print(f"[DEBUG] expand translation: tx={rot_tx:.3f}, ty={rot_ty:.3f}")

    # --- 4) tight crop to rotated mask bbox (+ padding) ---
    bbox = mask_rot.getbbox()
    if bbox is None:
        raise RuntimeError("Mask bbox empty after rotation. Check AOI vs GeoTIFF extent.")

    x0, y0, x1, y1 = bbox

    # fraction-based bbox padding (post-rotation)
    frac = float(clamp(args.crop_pad_frac, 0.0, 2.0))
    if frac > 0:
        bw = x1 - x0
        bh = y1 - y0
        fx = int(round(bw * frac))
        fy = int(round(bh * frac))
        x0 -= fx
        x1 += fx
        y0 -= fy
        y1 += fy

    crop_pad = max(0, int(args.crop_pad_px))
    x0 = max(0, x0 - crop_pad)
    y0 = max(0, y0 - crop_pad)
    x1 = min(Wr, x1 + crop_pad)
    y1 = min(Hr, y1 + crop_pad)

    hill_out = hill_rot.crop((x0, y0, x1, y1)).convert("RGBA")
    mask_out_L = mask_rot.crop((x0, y0, x1, y1))
    preW, preH = hill_out.size

    # --- 5) bake paper look (does not affect geometry) ---
    hill_out = apply_paper_look_rgba(
        hill_out,
        hill_alpha=args.hill_alpha,
        hill_gamma=args.hill_gamma,
        paper_rgb=paper,
        exposure=args.exposure,
    )

    # --- 6) optional resize (geometry changes; store scale factors) ---
    outW, outH = hill_out.size
    squished = bool(args.squish)
    if squished:
        tw = int(args.target_w)
        th = int(args.target_h)
        hill_out = hill_out.resize((tw, th), resample=Image.BICUBIC)
        mask_out_L = mask_out_L.resize((tw, th), resample=Image.NEAREST)
        outW, outH = tw, th

    mask_out = Image.new("RGBA", mask_out_L.size, (255, 255, 255, 0))
    mask_out.putalpha(mask_out_L)

    # --- 7) write meta so you can map final pixels back to lon/lat ---
    meta = {
        "geotiff": {
            "lon0": lon0,
            "lat0": lat0,
            "dlon": dlon,
            "dlat": dlat,
            "src_W": W,
            "src_H": H,
        },
        "pixel_transforms": {
            "rot_deg": rot_deg,
            "rot_rad": rot_rad,
            "rot_canvas_W": Wr,
            "rot_canvas_H": Hr,
            "rot_tx": rot_tx,
            "rot_ty": rot_ty,
            "src_center_x": src_cx,
            "src_center_y": src_cy,
            "crop_x0": int(x0),
            "crop_y0": int(y0),
            "pre_squish_W": int(preW),
            "pre_squish_H": int(preH),
            "out_W": int(outW),
            "out_H": int(outH),
            "squished": squished,
        },
        "notes": [
            "AOI lon/lat -> pixels uses GeoTIFF geotransform, so hillshade.png must match the GeoTIFF pixel grid.",
            "Paper look only changes colors, not pixel geometry.",
            "To map final pixel -> lon/lat: undo resize -> undo crop -> undo rotate+expand -> GeoTIFF pixel->lon/lat.",
        ],
    }

    out_hill = out_dir / args.out_hill
    out_mask = out_dir / args.out_mask
    out_meta = out_dir / args.out_meta

    hill_out.save(out_hill, "WEBP", quality=int(clamp(args.webp_quality, 0, 100)), method=6)
    mask_out.save(out_mask, "PNG")
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Wrote:")
    print(" -", out_hill)
    print(" -", out_mask)
    print(" -", out_meta)

if __name__ == "__main__":
    main()
