#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------- util ----------

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def load_geojson(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

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
        raise RuntimeError("AOI bounds look invalid.")
    return west, south, east, north

def lonlat_to_src_px(lon, lat, m):
    """
    lon/lat -> src pixel (pre-rotation).
    """
    west = float(m["bounds_west"]); south = float(m["bounds_south"])
    east = float(m["bounds_east"]); north = float(m["bounds_north"])
    src_W = float(m["src_W"]); src_H = float(m["src_H"])

    x = (lon - west) / (east - west) * src_W
    y = (north - lat) / (north - south) * src_H
    return x, y

def lonlat_to_out_px(lon, lat, m):
    """
    lon/lat -> src px -> rotate(+ang) -> crop -> squish
    Must match bake.py output assets.
    """
    x, y = lonlat_to_src_px(lon, lat, m)

    ang = float(m["rot_rad"])
    cx0 = float(m["src_center_x"]); cy0 = float(m["src_center_y"])
    cxr = float(m["rot_center_x"]); cyr = float(m["rot_center_y"])

    dx = x - cx0
    dy = y - cy0

    # forward rotation (same direction as PIL rotate(+deg))
    xr =  math.cos(ang)*dx - math.sin(ang)*dy + cxr
    yr =  math.sin(ang)*dx + math.cos(ang)*dy + cyr

    # crop into rot-crop space
    xr -= float(m["crop_x0"])
    yr -= float(m["crop_y0"])

    # squish (resize) rot-crop -> out
    if bool(m.get("squished", False)):
        preW = float(m["pre_squish_W"])
        preH = float(m["pre_squish_H"])
        out_W = float(m["out_W"])
        out_H = float(m["out_H"])
        sx = out_W / preW
        sy = out_H / preH
        xr *= sx
        yr *= sy

    return xr, yr

def draw_polyline(draw, pts, width=5, fill=(255, 220, 80, 255)):
    if len(pts) >= 2:
        draw.line(pts, fill=fill, width=width)

def draw_point(draw, x, y, label, font, color=(255, 60, 60, 255)):
    r = 7
    draw.ellipse((x-r, y-r, x+r, y+r), outline=color, width=3)
    draw.text((x+10, y-10), label, fill=color, font=font)

def apply_alpha(img_rgba: Image.Image, alpha_L: Image.Image) -> Image.Image:
    out = img_rgba.copy()
    out.putalpha(alpha_L)
    return out

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Debug axes in BOTH spaces: src (pre-rotation) and out (post-rot+crop+squish)."
    )
    ap.add_argument("--meta", required=True, help="bounds_rotcrop.json from bake.py")
    ap.add_argument("--hill-out", required=True, help="hillshade_paper_rotcrop.webp (final)")
    ap.add_argument("--hill-src", required=True, help="hillshade.png (pre-rotation)")
    ap.add_argument("--aoi", required=True, help="aoi_localidades.geojson (same used in bake.py)")
    ap.add_argument("--pad-frac", type=float, default=0.08, help="Must match bake.py pad-frac used to produce meta.")
    ap.add_argument("--out-out", required=True, help="Output debug image in OUT space")
    ap.add_argument("--out-src", required=True, help="Output debug image in SRC space")
    args = ap.parse_args()

    meta = load_json(Path(args.meta))
    m = meta["mapping"] if "mapping" in meta else meta

    # Load fonts
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()

    # Points you care about
    pts = [
        ("EL DORADO",      -74.075833, 4.598056),
        ("Monserrate",     -74.055500, 4.605700),
        ("Guadalupe",      -74.054400, 4.591900),
        ("Alto de la Viga",-74.035600, 4.574700),
        ("Cerro Aguanoso", -74.054700, 4.577500),
    ]

    # ----- compute the SAME mid_lon and bounds range used by bake.py -----
    # bake.py bounds = padded AOI bbox by pad_frac
    aoi_fc = load_geojson(Path(args.aoi))
    west0, south0, east0, north0 = get_bounds_from_aoi(aoi_fc)
    lonR = east0 - west0
    latR = north0 - south0
    pad = float(args.pad_frac)
    bounds = (
        west0 - lonR * pad,
        south0 - latR * pad,
        east0 + lonR * pad,
        north0 + latR * pad,
    )
    bounds_west, bounds_south, bounds_east, bounds_north = bounds
    mid_lon = 0.5 * (bounds_west + bounds_east)

    # Sanity: these should match meta bounds extremely closely
    # (if they don't, your pad-frac or AOI changed)
    # We don't hard error; we label it.
    bw = float(m["bounds_west"]); bs = float(m["bounds_south"])
    be = float(m["bounds_east"]); bn = float(m["bounds_north"])
    bounds_diff = max(abs(bounds_west-bw), abs(bounds_south-bs), abs(bounds_east-be), abs(bounds_north-bn))

    # ========== OUT SPACE DEBUG ==========
    out_img = Image.open(args.hill_out).convert("RGBA")
    d_out = ImageDraw.Draw(out_img)

    # Big north-south line in OUT space: constant lon = mid_lon, lat from bounds_north->bounds_south
    N = 400
    ns_out = []
    for i in range(N):
        t = i / (N - 1)
        lat = bounds_north + (bounds_south - bounds_north) * t
        x, y = lonlat_to_out_px(mid_lon, lat, m)
        ns_out.append((x, y))
    draw_polyline(d_out, ns_out, width=6, fill=(255, 220, 80, 255))
    d_out.text((ns_out[0][0] + 10, ns_out[0][1] + 10), "N->S (constant lon)", fill=(255, 220, 80, 255), font=font)

    # Optional graticule: constant lat line too (helps sanity)
    mid_lat = 0.5 * (bounds_south + bounds_north)
    ew_out = []
    for i in range(N):
        t = i / (N - 1)
        lon = bounds_west + (bounds_east - bounds_west) * t
        x, y = lonlat_to_out_px(lon, mid_lat, m)
        ew_out.append((x, y))
    draw_polyline(d_out, ew_out, width=4, fill=(120, 255, 120, 255))
    d_out.text((ew_out[0][0] + 10, ew_out[0][1] + 10), "W->E (constant lat)", fill=(120, 255, 120, 255), font=font)

    # Draw points in OUT
    for name, lon, lat in pts:
        x, y = lonlat_to_out_px(lon, lat, m)
        draw_point(d_out, x, y, name, font)

    # Label bounds diff
    d_out.text((10, 10), f"bounds(meta vs recompute) max diff: {bounds_diff:.8f} deg", fill=(255,255,255,220), font=font)

    Path(args.out_out).parent.mkdir(parents=True, exist_ok=True)
    out_img.save(args.out_out)
    print("Wrote:", args.out_out)

    # ========== SRC SPACE DEBUG ==========
    src_img = Image.open(args.hill_src).convert("RGBA")
    d_src = ImageDraw.Draw(src_img)

    # Build AOI alpha mask in SRC pixel space (same as bake.py step before rotation)
    # We only need it for visualization; simplest: rasterize polygon(s) to L mask
    src_W, src_H = src_img.size
    aoi_mask = Image.new("L", (src_W, src_H), 0)
    md = ImageDraw.Draw(aoi_mask)

    def lonlat_to_px_bounds(lon, lat):
        x = (lon - bounds_west) / (bounds_east - bounds_west) * src_W
        y = (bounds_north - lat) / (bounds_north - bounds_south) * src_H
        return (x, y)

    def draw_polygon_with_holes(rings_px):
        if not rings_px:
            return
        outer = rings_px[0]
        if len(outer) >= 3:
            md.polygon(outer, fill=255)
        for hole in rings_px[1:]:
            if len(hole) >= 3:
                md.polygon(hole, fill=0)

    for feat in aoi_fc.get("features", []):
        g = feat.get("geometry")
        if not g:
            continue
        t = g.get("type")
        coords = g.get("coordinates", [])
        if t == "Polygon":
            rings_px = []
            for ring in coords:
                rings_px.append([lonlat_to_px_bounds(lon, lat) for lon, lat in ring])
            draw_polygon_with_holes(rings_px)
        elif t == "MultiPolygon":
            for poly in coords:
                rings_px = []
                for ring in poly:
                    rings_px.append([lonlat_to_px_bounds(lon, lat) for lon, lat in ring])
                draw_polygon_with_holes(rings_px)

    src_img = apply_alpha(src_img, aoi_mask)
    d_src = ImageDraw.Draw(src_img)

    # Big north-south line in SRC space (should look vertical-ish because it's constant lon)
    ns_src = []
    for i in range(N):
        t = i / (N - 1)
        lat = bounds_north + (bounds_south - bounds_north) * t
        x, y = lonlat_to_px_bounds(mid_lon, lat)
        ns_src.append((x, y))
    draw_polyline(d_src, ns_src, width=6, fill=(255, 220, 80, 255))
    d_src.text((ns_src[0][0] + 10, ns_src[0][1] + 10), "N->S (constant lon) in SRC", fill=(255, 220, 80, 255), font=font)

    # Draw points in SRC (no rotation)
    for name, lon, lat in pts:
        x, y = lonlat_to_px_bounds(lon, lat)
        draw_point(d_src, x, y, name, font)

    Path(args.out_src).parent.mkdir(parents=True, exist_ok=True)
    src_img.save(args.out_src)
    print("Wrote:", args.out_src)

if __name__ == "__main__":
    main()
