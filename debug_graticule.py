#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def lonlat_to_out_px(lon, lat, m):
    """
    lon/lat -> src px -> rotate(+ang) -> crop -> squish(scale to out)
    Must match the assets baked by bake.py.
    """
    west = float(m["bounds_west"]); south = float(m["bounds_south"])
    east = float(m["bounds_east"]); north = float(m["bounds_north"])
    src_W = float(m["src_W"]); src_H = float(m["src_H"])

    # 1) lon/lat -> src pixel
    x = (lon - west) / (east - west) * src_W
    y = (north - lat) / (north - south) * src_H

    # 2) src -> rotated canvas (PIL rotate(+deg) convention, using same math as verify_points)
    ang = float(m["rot_rad"])
    cx0 = float(m["src_center_x"]); cy0 = float(m["src_center_y"])
    cxr = float(m["rot_center_x"]); cyr = float(m["rot_center_y"])

    dx = x - cx0
    dy = y - cy0

    xr =  math.cos(ang)*dx - math.sin(ang)*dy + cxr
    yr =  math.sin(ang)*dx + math.cos(ang)*dy + cyr

    # 3) crop into rot-crop space
    xr -= float(m["crop_x0"])
    yr -= float(m["crop_y0"])

    # 4) squish (resize) rot-crop -> out
    out_W = float(m["out_W"]); out_H = float(m["out_H"])
    if bool(m.get("squished", False)):
        preW = float(m["pre_squish_W"])
        preH = float(m["pre_squish_H"])
        sx = out_W / preW
        sy = out_H / preH
        xr *= sx
        yr *= sy

    return xr, yr

def draw_polyline(draw, pts, width=3, fill=(255, 80, 80, 255)):
    if len(pts) >= 2:
        draw.line(pts, fill=fill, width=width)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--hill", required=True, help="hillshade_paper_rotcrop.webp")
    ap.add_argument("--out", required=True, help="debug_graticule.png")
    args = ap.parse_args()

    meta = load_json(Path(args.meta))
    m = meta["mapping"] if "mapping" in meta else meta

    img = Image.open(args.hill).convert("RGBA")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()

    # Your reference points
    pts = [
        ("EL DORADO",     -74.075833, 4.598056),
        ("Monserrate",    -74.055500, 4.605700),
        ("Guadalupe",     -74.054400, 4.591900),
        ("Alto de la Viga",-74.035600, 4.574700),
        ("Cerro Aguanoso",-74.054700, 4.577500),
    ]

    # Draw points
    for name, lon, lat in pts:
        x, y = lonlat_to_out_px(lon, lat, m)
        r = 6
        draw.ellipse((x-r, y-r, x+r, y+r), outline=(255, 60, 60, 255), width=3)
        draw.text((x+10, y-10), name, fill=(255, 60, 60, 255), font=font)

    # Draw graticule: one constant-lon line and one constant-lat line
    west = float(m["bounds_west"]); east = float(m["bounds_east"])
    south = float(m["bounds_south"]); north = float(m["bounds_north"])

    mid_lon = 0.5 * (west + east)
    mid_lat = 0.5 * (south + north)

    N = 200
    # constant lon (north->south)
    lon_line = []
    for i in range(N):
        t = i / (N - 1)
        lat = north + (south - north) * t
        x, y = lonlat_to_out_px(mid_lon, lat, m)
        lon_line.append((x, y))
    draw_polyline(draw, lon_line, width=3, fill=(80, 200, 255, 255))
    draw.text((lon_line[0][0]+8, lon_line[0][1]+8), "constant lon", fill=(80, 200, 255, 255), font=font)

    # constant lat (west->east)
    lat_line = []
    for i in range(N):
        t = i / (N - 1)
        lon = west + (east - west) * t
        x, y = lonlat_to_out_px(lon, mid_lat, m)
        lat_line.append((x, y))
    draw_polyline(draw, lat_line, width=3, fill=(120, 255, 120, 255))
    draw.text((lat_line[0][0]+8, lat_line[0][1]+8), "constant lat", fill=(120, 255, 120, 255), font=font)

    # Draw "north arrow" at center: take a point and the same lon with +delta lat
    base_lon, base_lat = mid_lon, mid_lat
    dlat = (north - south) * 0.03  # 3% of bounds height
    x0, y0 = lonlat_to_out_px(base_lon, base_lat, m)
    x1, y1 = lonlat_to_out_px(base_lon, base_lat + dlat, m)

    draw.line([(x0, y0), (x1, y1)], fill=(255, 220, 80, 255), width=5)
    draw.ellipse((x0-4, y0-4, x0+4, y0+4), fill=(255, 220, 80, 255))
    draw.text((x1+8, y1-8), "N", fill=(255, 220, 80, 255), font=font)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print("Wrote:", args.out)

if __name__ == "__main__":
    main()
