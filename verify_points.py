#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def unpack_mapping(meta):
    if "mapping" in meta:
        m = meta["mapping"]
    else:
        m = meta
    return m


def lonlat_to_outpx(m, lon, lat):
    """
    Forward mapping using PIL's actual expand translation:
      lon/lat -> src px -> rotate around src center -> apply PIL translation -> crop -> squish
    """
    west = float(m["bounds_west"])
    south = float(m["bounds_south"])
    east = float(m["bounds_east"])
    north = float(m["bounds_north"])
    src_W = float(m["src_W"])
    src_H = float(m["src_H"])

    # 1) lon/lat -> src pixel
    src_x = (lon - west) / (east - west) * src_W
    src_y = (north - lat) / (north - south) * src_H

    # 2) rotate around src center
    ang = float(m["rot_rad"])
    cosA = math.cos(ang)
    sinA = math.sin(ang)
    
    src_cx = src_W / 2.0
    src_cy = src_H / 2.0
    
    dx = src_x - src_cx
    dy = src_y - src_cy
    
    # Pure rotation
    rx = cosA * dx - sinA * dy
    ry = sinA * dx + cosA * dy
    
    # 3) Apply PIL's expand translation
    rot_tx = float(m["rot_tx"])
    rot_ty = float(m["rot_ty"])
    
    rot_x = rx + rot_tx
    rot_y = ry + rot_ty

    # 4) Crop
    crop_x = rot_x - float(m["crop_x0"])
    crop_y = rot_y - float(m["crop_y0"])

    # 5) Squish
    if bool(m.get("squished", False)):
        preW = float(m["pre_squish_W"])
        preH = float(m["pre_squish_H"])
        out_W = float(m["out_W"])
        out_H = float(m["out_H"])
        sx = out_W / preW
        sy = out_H / preH
        out_x = crop_x * sx
        out_y = crop_y * sy
    else:
        out_x = crop_x
        out_y = crop_y

    return out_x, out_y


def decode_elev_rg(elev_rgba, elev_min, elev_max, x, y):
    """
    Nearest-neighbor sample at integer pixel (x,y).
    elev_rgba: uint8 HxWx4
    """
    H, W, _ = elev_rgba.shape
    if x < 0 or x >= W or y < 0 or y >= H:
        return None, None, None

    R = int(elev_rgba[y, x, 0])
    G = int(elev_rgba[y, x, 1])
    A = int(elev_rgba[y, x, 3])

    v = R * 256 + G  # 0..65535
    elev = elev_min + (v / 65535.0) * (elev_max - elev_min)
    return elev, v, A


def main():
    ap = argparse.ArgumentParser(description="Verify baked elevation against known points.")
    ap.add_argument("--meta", required=True, help="bounds_rotcrop.json")
    ap.add_argument("--elev-meta", required=True, help="elev_meta.json")
    ap.add_argument("--elev-rg", required=True, help="elev_rg_rotcrop.png")
    args = ap.parse_args()

    m = unpack_mapping(load_json(Path(args.meta)))
    em = load_json(Path(args.elev_meta))

    elev_min = float(em["elev_min_m"])
    elev_max = float(em["elev_max_m"])

    img = Image.open(Path(args.elev_rg)).convert("RGBA")
    elev_rgba = np.array(img, dtype=np.uint8)
    H, W, _ = elev_rgba.shape

    points = [
        ("EL DORADO approx", -74.075833, 4.598056, 2613.0),   # 4°35′53″N 74°04′33″W
        ("Monserrate",      -74.0555,   4.6057,   3152.0),
        ("Guadalupe",       -74.0544,   4.5919,   3260.0),
        ("Alto de la Viga", -74.0356,   4.5747,   3648.0),
        ("Cerro Aguanoso",  -74.0547,   4.5775,   3450.0),
    ]

    print(f"Image size: {W}x{H}")
    print(f"Elevation decode range: {elev_min:.3f} .. {elev_max:.3f} m")
    print()

    for name, lon, lat, expected in points:
        out_x, out_y = lonlat_to_outpx(m, lon, lat)
        xi = int(round(out_x))
        yi = int(round(out_y))

        elev, v, a = decode_elev_rg(elev_rgba, elev_min, elev_max, xi, yi)

        print(f"{name}")
        print(f"  lon,lat: {lon:.6f}, {lat:.6f}")
        print(f"  out px : ({out_x:.2f}, {out_y:.2f}) -> nearest ({xi}, {yi})")

        if elev is None:
            print("  RESULT: outside image bounds")
            print()
            continue

        inside = (a > 0)
        err = elev - expected
        print(f"  alpha  : {a}  (inside AOI: {inside})")
        print(f"  elev   : {elev:.1f} m   (expected ~{expected:.1f} m, diff {err:+.1f} m)")
        print()

if __name__ == "__main__":
    main()
