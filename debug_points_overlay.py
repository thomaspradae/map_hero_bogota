#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def lonlat_to_out_px(lon, lat, m):
    # This must match your verify_points.py mapping.
    # We go lon/lat -> src px -> rotate -> crop -> squish.
    west = m["bounds_west"]; south = m["bounds_south"]; east = m["bounds_east"]; north = m["bounds_north"]
    src_W = m["src_W"]; src_H = m["src_H"]

    # lon/lat -> src pixel
    x = (lon - west) / (east - west) * src_W
    y = (north - lat) / (north - south) * src_H

    # src -> rotated canvas
    ang = m["rot_rad"]
    cx0 = m["src_center_x"]; cy0 = m["src_center_y"]
    cxr = m["rot_center_x"]; cyr = m["rot_center_y"]

    dx = x - cx0
    dy = y - cy0

    # forward rotation (same direction as PIL rotate(+deg))
    xr =  math.cos(ang)*dx - math.sin(ang)*dy + cxr
    yr =  math.sin(ang)*dx + math.cos(ang)*dy + cyr

    # crop
    xr -= m["crop_x0"]
    yr -= m["crop_y0"]

    # squish (crop size -> out size)
    out_W = m["out_W"]; out_H = m["out_H"]
    # If you saved out_W/out_H after squish, then xr/yr are already in out space only if you squished by resize.
    # But your pipeline did resize AFTER crop, so we scale:
    # crop_W = rot_crop_W, crop_H = rot_crop_H are not stored; derive scale from target_W/H if squished.
    if m.get("squished") and m.get("target_W") and m.get("target_H"):
        # out_W == target_W, out_H == target_H
        # We need crop dimensions before resize; not stored, but we can compute them from out and scale if you stored them.
        # If you didn't store cropW/cropH, just treat xr/yr as already out-space only if you do NOT resize.
        # Practical fix: store cropW/cropH in meta next time.
        pass

    return xr, yr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--hill", required=True, help="hillshade_paper_rotcrop.webp")
    ap.add_argument("--out", required=True, help="debug_points.png")
    args = ap.parse_args()

    meta = load_json(Path(args.meta))["mapping"]
    img = Image.open(args.hill).convert("RGBA")
    draw = ImageDraw.Draw(img)

    pts = [
        ("EL DORADO", -74.075833, 4.598056),
        ("Monserrate", -74.055500, 4.605700),
        ("Guadalupe", -74.054400, 4.591900),
        ("Alto de la Viga", -74.035600, 4.574700),
        ("Cerro Aguanoso", -74.054700, 4.577500),
    ]

    # basic font fallback
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()

    for name, lon, lat in pts:
        x, y = lonlat_to_out_px(lon, lat, meta)
        r = 6
        draw.ellipse((x-r, y-r, x+r, y+r), outline=(255, 50, 50, 255), width=3)
        draw.text((x+10, y-10), name, fill=(255, 50, 50, 255), font=font)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print("Wrote:", args.out)

if __name__ == "__main__":
    main()
