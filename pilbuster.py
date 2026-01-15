import numpy as np
from PIL import Image

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
    print("  rotated canvas size (PIL) =", Image.new("L",(W,H),0).rotate(ROT_DEG, Image.NEAREST, expand=True).size)
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
