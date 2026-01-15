#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import rasterio
except ImportError as e:
    raise SystemExit(
        "Missing rasterio. Install it in your venv:\n"
        "  pip install rasterio\n"
        f"Original error: {e}"
    )

# ---------------- util ----------------

def clamp(x, a, b):
    return max(a, min(b, x))

def load_geojson(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def lonlat_to_px_geotiff(lon, lat, lon0, lat0, dlon, dlat):
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
    """
    Laplacian smoothing on a closed ring (matches TS laplacianSmooth2D).
    pts: list of (x,y) tuples, assumed closed (first == last)
    """
    if passes <= 0 or len(pts) < 4:
        return pts
    curr = np.array(pts, dtype=np.float32)
    n = len(curr)

    for _ in range(passes):
        nxt = curr.copy()
        for i in range(n - 1):  # ignore last point for updates, mirror to close later
            prev_i = (i - 1) if i > 0 else (n - 2)
            next_i = (i + 1) if i < (n - 2) else 0
            avg = 0.5 * (curr[prev_i] + curr[next_i])
            nxt[i] = (1 - alpha) * curr[i] + alpha * avg
        nxt[n - 1] = nxt[0]  # keep closed
        curr = nxt
    return [tuple(p) for p in curr]

def scale_ring_about_centroid(pts, s):
    """
    Scale ring about its centroid (matches TS clipScale behavior).
    """
    if abs(s - 1.0) < 1e-6 or len(pts) < 3:
        return pts
    arr = np.array(pts, dtype=np.float32)
    cx, cy = arr[:, 0].mean(), arr[:, 1].mean()
    arr[:, 0] = cx + (arr[:, 0] - cx) * s
    arr[:, 1] = cy + (arr[:, 1] - cy) * s
    return [tuple(p) for p in arr]

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

# ---------------- "paper look" bake ----------------
def apply_paper_look_rgba(img_rgba: Image.Image, hill_alpha: float, hill_gamma: float, paper_rgb, exposure: float):
    """
    Mimics your WebGL fragment shader:
      hl = lum(hill)  (your lum() is average)
      hl = pow(hl, 0.85)
      base = mix(paperBase, vec3(hl), HILL_ALPHA)
    We apply it per-pixel to RGB, preserving alpha.
    Then apply exposure multiplier for final brightness control.
    """
    hill_alpha = float(clamp(hill_alpha, 0.0, 1.0))
    hill_gamma = max(1e-6, float(hill_gamma))
    exposure = float(exposure)
    paper = np.array(paper_rgb, dtype=np.float32)  # 0..1

    arr = np.array(img_rgba, dtype=np.uint8)  # H,W,4
    rgb = arr[..., :3].astype(np.float32) / 255.0
    a = arr[..., 3:4].copy()

    # average luminance (matches your vec3(0.3333))
    hl = (rgb[..., 0] + rgb[..., 1] + rgb[..., 2]) / 3.0
    hl = np.clip(hl, 0.0, 1.0)
    hl = np.power(hl, hill_gamma)  # NOTE: you use pow(hl,0.85) so gamma=0.85

    # mix paper and grayscale hl
    # base = (1-alpha)*paper + alpha*hl
    base = (1.0 - hill_alpha) * paper[None, None, :] + hill_alpha * hl[..., None]
    base = np.clip(base * exposure, 0.0, 1.0)

    out = np.concatenate([(base * 255.0 + 0.5).astype(np.uint8), a], axis=-1)
    return Image.fromarray(out, mode="RGBA")

# ---------------- main pipeline ----------------

def main():
    ap = argparse.ArgumentParser(
        description="AOI alpha -> rotate -> tight-crop -> bake WebGL paper look -> optional squish to 910x820."
    )
    ap.add_argument("--aoi", required=True, help="Path to AOI GeoJSON (aoi_localidades.geojson).")
    ap.add_argument("--hillshade", required=True, help="Path to hillshade image (hillshade.png).")
    ap.add_argument("--geotiff", required=True, help="Path to source GeoTIFF (e.g. bogota_raw_srtm.tif) for geotransform.")
    ap.add_argument("--outdir", required=True, help="Output directory (e.g. .../public/bogota).")

    # rotation and crop
    ap.add_argument("--rot-deg", type=float, default=120.0, help="Rotation degrees to bake into output assets.")
    ap.add_argument("--crop-pad-px", type=int, default=8, help="Extra pixel padding around tight crop bbox (post-rotation).")

    # paper look knobs (match shader)
    ap.add_argument("--hill-alpha", type=float, default=0.30, help="HILL_ALPHA from shader (0..1).")
    ap.add_argument("--hill-gamma", type=float, default=0.85, help="pow(hl, gamma) from shader.")
    ap.add_argument("--paper", type=str, default="0.975,0.972,0.965", help="paperBase RGB in 0..1, comma-separated.")
    ap.add_argument("--exposure", type=float, default=1.0, help="Final brightness multiplier after paper mix (e.g. 0.95 = 5% darker).")

    # AOI conditioning (match TS GPU pipeline)
    ap.add_argument("--clip-smooth-passes", type=int, default=0, help="Laplacian smoothing passes on AOI rings (match TS: 8).")
    ap.add_argument("--clip-scale", type=float, default=1.0, help="Scale AOI rings about centroid (match TS: 1.08).")

    # “same squish” (match inner rect)
    ap.add_argument("--squish", action="store_true", help="Resize final hillshade+mask to target WxH (default 910x820).")
    ap.add_argument("--target-w", type=int, default=910, help="Target width if --squish.")
    ap.add_argument("--target-h", type=int, default=820, help="Target height if --squish.")

    # outputs
    ap.add_argument("--out-hill", type=str, default="hillshade_paper_rotcrop.webp", help="Output hillshade filename.")
    ap.add_argument("--out-mask", type=str, default="aoi_mask_rotcrop.png", help="Output AOI mask filename.")
    ap.add_argument("--out-meta", type=str, default="bounds_rotcrop.json", help="Output meta filename.")
    ap.add_argument("--webp-quality", type=int, default=92, help="WebP quality (0..100).")

    args = ap.parse_args()

    aoi_path = Path(args.aoi).expanduser().resolve()
    hill_path = Path(args.hillshade).expanduser().resolve()
    geotiff_path = Path(args.geotiff).expanduser().resolve()
    out_dir = Path(args.outdir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI not found: {aoi_path}")
    if not hill_path.exists():
        raise FileNotFoundError(f"Hillshade not found: {hill_path}")
    if not geotiff_path.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {geotiff_path}")

    rot_deg = float(args.rot_deg)
    rot_rad = rot_deg * math.pi / 180.0
    crop_pad = max(0, int(args.crop_pad_px))

    paper = tuple(float(x) for x in args.paper.split(","))
    if len(paper) != 3:
        raise ValueError("--paper must be 'r,g,b' in 0..1")

    # --- 1) Read GeoTIFF geotransform ---
    with rasterio.open(geotiff_path) as ds:
        transform = ds.transform
        # For north-up rasters: transform is [dlon, 0, lon0, 0, dlat, lat0]
        # where dlon > 0, dlat < 0 typically
        lon0 = float(transform.c)  # x origin (longitude at pixel 0,0)
        lat0 = float(transform.f)  # y origin (latitude at pixel 0,0)
        dlon = float(transform.a)   # pixel width in longitude
        dlat = float(transform.e)   # pixel height in latitude (usually negative)
        
        geotiff_W = ds.width
        geotiff_H = ds.height
        
        print(f"[DEBUG] GeoTIFF geotransform: lon0={lon0:.8f}, lat0={lat0:.8f}, dlon={dlon:.10f}, dlat={dlat:.10f}")
        print(f"[DEBUG] GeoTIFF size: {geotiff_W}x{geotiff_H}")

    # --- 2) load hillshade (RGBA) ---
    hill = Image.open(hill_path).convert("RGBA")
    W, H = hill.size
    
    if W != geotiff_W or H != geotiff_H:
        raise RuntimeError(
            f"Hillshade size {W}x{H} does not match GeoTIFF size {geotiff_W}x{geotiff_H}. "
            "Hillshade must be aligned 1:1 with GeoTIFF pixel grid."
        )

    # --- 3) load AOI and build alpha mask in hillshade pixel space ---
    fc = load_geojson(aoi_path)
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
                # Use GeoTIFF geotransform to convert lon/lat -> pixel
                ring_px = [lonlat_to_px_geotiff(lon, lat, lon0, lat0, dlon, dlat) for lon, lat in ring]
                # ensure closed
                if len(ring_px) >= 3 and (abs(ring_px[0][0] - ring_px[-1][0]) > 1e-3 or abs(ring_px[0][1] - ring_px[-1][1]) > 1e-3):
                    ring_px = ring_px + [ring_px[0]]
                # smooth + scale (to match TS)
                sp = max(0, int(args.clip_smooth_passes))
                if sp > 0:
                    ring_px = laplacian_smooth_ring(ring_px, sp, alpha=0.5)
                cs = float(args.clip_scale)
                if cs != 1.0:
                    ring_px = scale_ring_about_centroid(ring_px, cs)
                rings_px.append(ring_px)
            draw_polygon_with_holes(draw, rings_px)

        elif t == "MultiPolygon":
            for poly in coords:
                rings_px = []
                for ring in poly:
                    # Use GeoTIFF geotransform to convert lon/lat -> pixel
                    ring_px = [lonlat_to_px_geotiff(lon, lat, lon0, lat0, dlon, dlat) for lon, lat in ring]
                    # ensure closed
                    if len(ring_px) >= 3 and (abs(ring_px[0][0] - ring_px[-1][0]) > 1e-3 or abs(ring_px[0][1] - ring_px[-1][1]) > 1e-3):
                        ring_px = ring_px + [ring_px[0]]
                    # smooth + scale (to match TS)
                    sp = max(0, int(args.clip_smooth_passes))
                    if sp > 0:
                        ring_px = laplacian_smooth_ring(ring_px, sp, alpha=0.5)
                    cs = float(args.clip_scale)
                    if cs != 1.0:
                        ring_px = scale_ring_about_centroid(ring_px, cs)
                    rings_px.append(ring_px)
                draw_polygon_with_holes(draw, rings_px)

    # Apply alpha
    hill.putalpha(mask)

    # --- 4) rotate BOTH (expand canvas) ---
    mask_rot = mask.rotate(rot_deg, resample=Image.NEAREST, expand=True, fillcolor=0)
    hill_rot = hill.rotate(rot_deg, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0))
    Wr, Hr = hill_rot.size

    # --- COMPUTE PIL's HIDDEN TRANSLATION ---
    # PIL rotate(expand=True) does: rotate around center, then translate so bbox starts at (0,0)
    # We measure this translation directly by tracking where src center lands
    src_cx = W / 2.0
    src_cy = H / 2.0

    # Pure rotation of center point (no translation yet)
    cosA = math.cos(rot_rad)
    sinA = math.sin(rot_rad)
    # Rotate (0,0) displacement → still at origin after pure rotation
    rot_pure_cx = src_cx  # will be at center of SOME coordinate system
    rot_pure_cy = src_cy

    # Where PIL actually puts the center in the expanded canvas
    pil_canvas_cx = Wr / 2.0
    pil_canvas_cy = Hr / 2.0

    # The translation PIL applied = where center ended up - where pure rotation would put it
    # For a pure rotation around image center, center stays at center
    # But PIL's expand shifts everything so the bounding box min corner is at (0,0)
    # We can measure this by checking where any known point lands

    # Better approach: compute the translation directly from corner positions
    corners_src = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float64)
    corners_rot = np.zeros_like(corners_src)

    for i, (x, y) in enumerate(corners_src):
        dx = x - src_cx
        dy = y - src_cy
        rx = cosA * dx - sinA * dy
        ry = sinA * dx + cosA * dy
        corners_rot[i] = [rx, ry]

    # Min corner of rotated bbox (in rotation-centered coordinates)
    min_x = corners_rot[:, 0].min()
    min_y = corners_rot[:, 1].min()

    # PIL translation to make min corner = (0, 0)
    rot_tx = -min_x + src_cx
    rot_ty = -min_y + src_cy

    print(f"[DEBUG] Computed PIL expand translation: tx={rot_tx:.2f}, ty={rot_ty:.2f}")
    print(f"[DEBUG] Rotated canvas size: {Wr}x{Hr}")

    # --- 5) tight crop to rotated mask bbox (+ crop_pad) ---
    bbox = mask_rot.getbbox()
    if bbox is None:
        raise RuntimeError("Mask bbox empty after rotation. Increase --pad-frac or check AOI geometry.")

    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - crop_pad)
    y0 = max(0, y0 - crop_pad)
    x1 = min(Wr, x1 + crop_pad)
    y1 = min(Hr, y1 + crop_pad)

    hill_out = hill_rot.crop((x0, y0, x1, y1)).convert("RGBA")
    mask_out_L = mask_rot.crop((x0, y0, x1, y1))
    preW, preH = hill_out.size

    # --- 6) bake paper look on RGB, keep alpha ---
    hill_out = apply_paper_look_rgba(
        hill_out,
        hill_alpha=args.hill_alpha,
        hill_gamma=args.hill_gamma,
        paper_rgb=paper,
        exposure=args.exposure,
    )

    # --- 7) optional “same squish” to inner aspect (910×820) ---
    if args.squish:
        tw = int(args.target_w)
        th = int(args.target_h)

        hill_out = hill_out.resize((tw, th), resample=Image.BICUBIC)
        # mask should stay crisp => nearest
        mask_out_L = mask_out_L.resize((tw, th), resample=Image.NEAREST)

    # Build mask RGBA (white with alpha)
    mask_out = Image.new("RGBA", mask_out_L.size, (255, 255, 255, 0))
    mask_out.putalpha(mask_out_L)

    # --- 8) meta: mapping for pixel->lon/lat using GeoTIFF geotransform ---
    c_orig_x = (W - 1) * 0.5
    c_orig_y = (H - 1) * 0.5

    outW, outH = hill_out.size

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
            "rot_tx": float(rot_tx),
            "rot_ty": float(rot_ty),
            "src_center_x": c_orig_x,
            "src_center_y": c_orig_y,
            "crop_x0": int(x0),
            "crop_y0": int(y0),
            "pre_squish_W": int(preW),
            "pre_squish_H": int(preH),
            "out_W": int(outW),
            "out_H": int(outH),
            "squished": bool(args.squish),
            "target_W": int(args.target_w) if args.squish else None,
            "target_H": int(args.target_h) if args.squish else None,
        },
        "visual": {
            "paper_base": list(paper),
            "hill_alpha": float(args.hill_alpha),
            "hill_gamma": float(args.hill_gamma),
            "exposure": float(args.exposure),
            "clip_smooth_passes": int(args.clip_smooth_passes),
            "clip_scale": float(args.clip_scale),
        },
        "inputs": {
            "aoi": str(aoi_path),
            "hillshade": str(hill_path),
            "geotiff": str(geotiff_path),
            "crop_pad_px": int(crop_pad),
        },
        "notes": [
            "AOI lon/lat -> pixels uses GeoTIFF geotransform, so hillshade.png must match the GeoTIFF pixel grid.",
            "hillshade_paper_rotcrop already includes the WebGL 'paper mix' look.",
            "If squished=true, image was resized to inner rect (910x820 by default) to match original aspect.",
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
