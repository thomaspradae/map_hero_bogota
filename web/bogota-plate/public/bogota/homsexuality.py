#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def lonlat_to_final_uv(lon, lat, meta):
    g = meta["geotiff"]
    t = meta["pixel_transforms"]

    lon0, lat0, dlon, dlat = g["lon0"], g["lat0"], g["dlon"], g["dlat"]

    # 1) lon/lat -> src pixel (GeoTIFF grid)
    x = (lon - lon0) / dlon
    y = (lat - lat0) / dlat

    # 2) rotate around src center, then apply expand=True translation
    cx, cy = t["src_center_x"], t["src_center_y"]
    th = t["rot_rad"]
    cosA, sinA = math.cos(th), math.sin(th)

    dx, dy = x - cx, y - cy
    xr = cosA * dx - sinA * dy + cx + t["rot_tx"]
    yr = sinA * dx + cosA * dy + cy + t["rot_ty"]

    # 3) crop
    xc = xr - t["crop_x0"]
    yc = yr - t["crop_y0"]

    # 4) resize (if squished)
    if t["squished"]:
        sx = t["out_W"] / t["pre_squish_W"]
        sy = t["out_H"] / t["pre_squish_H"]
        u = xc * sx
        v = yc * sy
    else:
        u, v = xc, yc

    return u, v


def parse_point(s: str):
    # "Name,lon,lat"
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f'Invalid --point "{s}". Use "Name,lon,lat"')
    name = parts[0]
    lon = float(parts[1])
    lat = float(parts[2])
    return (name, lon, lat)


def main():
    ap = argparse.ArgumentParser(
        description="Plot lon/lat points onto the FINAL rotated/cropped/squished image using mapping_rotcrop.json."
    )
    ap.add_argument("--meta", required=True, help="Path to mapping_rotcrop.json")
    ap.add_argument("--image", required=True, help="Path to hillshade_paper_rotcrop.webp (or png)")
    ap.add_argument("--out", default="debug_points_on_final.png", help="Output annotated image path")

    ap.add_argument(
        "--point",
        action="append",
        default=[],
        help='Repeatable. Format: "Name,lon,lat" (lon/lat in degrees).',
    )
    ap.add_argument("--r", type=int, default=8, help="Crosshair radius in pixels")
    ap.add_argument("--label", action="store_true", help="Draw point labels")

    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    img = Image.open(args.image).convert("RGBA")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    # default points if none provided
    points = args.point
    if not points:
        points = [
            "Cerro Aguanoso,-74.0531,4.5756",
            "Alto de la Viga,-74.03185,4.5719",
            "Guadalupe Peak,-74.0544,4.5919",
            "Monserrate Peak,-74.0555,4.6057",
            "Cerro el Cable,-74.0507,4.6297",
        ]

    pts = [parse_point(s) for s in points]

    # font (optional; default bitmap font if available)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    r = int(args.r)
    for name, lon, lat in pts:
        u, v = lonlat_to_final_uv(lon, lat, meta)
        x, y = int(round(u)), int(round(v))
        inside = (0 <= x < W) and (0 <= y < H)
        print(f"{name:16s} -> u={u:.2f}, v={v:.2f}  (pix {x},{y})  inside={inside}")

        if inside:
            # crosshair
            draw.line([(x - r, y), (x + r, y)], fill=(255, 0, 0, 255), width=2)
            draw.line([(x, y - r), (x, y + r)], fill=(255, 0, 0, 255), width=2)

            # label
            if args.label:
                tx, ty = x + r + 4, y - r - 2
                # tiny backdrop for readability
                text = name
                if font:
                    bbox = draw.textbbox((tx, ty), text, font=font)
                    draw.rectangle(bbox, fill=(0, 0, 0, 140))
                    draw.text((tx, ty), text, fill=(255, 255, 255, 255), font=font)
                else:
                    draw.text((tx, ty), text, fill=(255, 255, 255, 255))

    out_path = Path(args.out)
    img.save(out_path)
    print("Wrote", out_path.resolve())


if __name__ == "__main__":
    main()
