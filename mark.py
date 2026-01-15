#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def dms_to_dd(s: str) -> float:
    """
    Parse strings like:
      4°34'32.5"N
      74°03'10.9"W
    Returns signed decimal degrees.
    """
    t = s.strip().replace(" ", "")
    m = re.match(r"""^(\d+)[°:](\d+)['](\d+(?:\.\d+)?)[\"]?([NSEW])$""", t, re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not parse DMS: {s}")
    deg = float(m.group(1))
    minu = float(m.group(2))
    sec = float(m.group(3))
    hemi = m.group(4).upper()
    dd = deg + minu / 60.0 + sec / 3600.0
    if hemi in ("S", "W"):
        dd = -dd
    return dd

def lonlat_to_src_px(lon: float, lat: float, m: dict):
    west = float(m["bounds_west"]); south = float(m["bounds_south"])
    east = float(m["bounds_east"]); north = float(m["bounds_north"])
    W = float(m["src_W"]); H = float(m["src_H"])

    x = (lon - west) / (east - west) * W
    y = (north - lat) / (north - south) * H
    return x, y

def main():
    ap = argparse.ArgumentParser(description="Put a marker (lon/lat) onto the ORIGINAL hillshade (SRC space).")
    ap.add_argument("--meta", required=True, help="bounds_rotcrop.json from bake.py")
    ap.add_argument("--hill-src", required=True, help="Original hillshade.png (pre-rotation)")
    ap.add_argument("--out", required=True, help="Output image path (e.g. debug_marker_src.png)")
    ap.add_argument("--lat", default='4°34\'32.5"N', help='Latitude DMS like 4°34\'32.5"N')
    ap.add_argument("--lon", default='74°03\'10.9"W', help='Longitude DMS like 74°03\'10.9"W')
    ap.add_argument("--label", default="DMS point", help="Label text")
    args = ap.parse_args()

    meta = load_json(Path(args.meta))
    m = meta["mapping"] if "mapping" in meta else meta

    lat = dms_to_dd(args.lat)
    lon = dms_to_dd(args.lon)

    img = Image.open(args.hill_src).convert("RGBA")
    W, H = img.size

    # sanity: meta src_W/H should match this src image
    if int(m["src_W"]) != W or int(m["src_H"]) != H:
        print(f"WARNING: meta src_W/H = {m['src_W']}x{m['src_H']} but hill-src is {W}x{H}")

    x, y = lonlat_to_src_px(lon, lat, m)

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()

    # marker
    r = 10
    draw.ellipse((x-r, y-r, x+r, y+r), outline=(255, 50, 50, 255), width=4)
    draw.line([(x-20, y), (x+20, y)], fill=(255, 50, 50, 255), width=3)
    draw.line([(x, y-20), (x, y+20)], fill=(255, 50, 50, 255), width=3)

    txt = f"{args.label}\nlat={lat:.9f}\nlon={lon:.9f}\npx=({x:.2f},{y:.2f})"
    draw.text((x + 14, y - 40), txt, fill=(255, 50, 50, 255), font=font)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print("Wrote:", args.out)
    print("Decimal degrees:", lat, lon)
    print("SRC px:", x, y)

if __name__ == "__main__":
    main()
