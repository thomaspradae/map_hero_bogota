#!/usr/bin/env python3
"""
Analyze calibration data to diagnose transform errors.
Compares actual clicked points vs where the current transform thinks they should be.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def lonlat_to_outpx_current(m, lon, lat):
    """
    Current (possibly broken) forward mapping from bounds_rotcrop.json
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
    
    rx = cosA * dx - sinA * dy
    ry = sinA * dx + cosA * dy
    
    # 3) Apply PIL's expand translation
    rot_tx = float(m.get("rot_tx", 0))
    rot_ty = float(m.get("rot_ty", 0))
    
    rot_x = rx + src_cx + rot_tx
    rot_y = ry + src_cy + rot_ty

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


def fit_affine_transform(src_points, dst_points):
    """
    Fit an affine transform from src to dst points.
    Returns 2x3 affine matrix that minimizes least-squares error.
    
    src_points: Nx2 array of (lon, lat)
    dst_points: Nx2 array of (x, y) pixels
    """
    # Solve: dst = src @ A.T + b
    # Rewrite as: dst = [src, 1] @ [A.T, b.T].T
    N = len(src_points)
    
    # Build augmented source matrix [lon, lat, 1]
    src_aug = np.column_stack([src_points, np.ones(N)])
    
    # Solve least squares for x and y separately
    # x = a*lon + b*lat + c
    # y = d*lon + e*lat + f
    
    coeffs_x, _, _, _ = np.linalg.lstsq(src_aug, dst_points[:, 0], rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(src_aug, dst_points[:, 1], rcond=None)
    
    # Build affine matrix [a, b, c; d, e, f]
    affine = np.array([
        [coeffs_x[0], coeffs_x[1], coeffs_x[2]],
        [coeffs_y[0], coeffs_y[1], coeffs_y[2]]
    ])
    
    return affine


def apply_affine(affine, points):
    """Apply affine transform to points."""
    N = len(points)
    pts_aug = np.column_stack([points, np.ones(N)])
    return pts_aug @ affine.T


def compute_residuals(src, dst, affine):
    """Compute residuals after applying affine transform."""
    pred = apply_affine(affine, src)
    residuals = dst - pred
    distances = np.linalg.norm(residuals, axis=1)
    return residuals, distances


def main():
    ap = argparse.ArgumentParser(
        description="Analyze calibration to diagnose coordinate transform errors"
    )
    ap.add_argument("--calibration", required=True, help="calibration.json from web tool")
    ap.add_argument("--meta", required=True, help="bounds_rotcrop.json from bake.py")
    ap.add_argument("--verbose", action="store_true", help="Show detailed diagnostics")
    
    args = ap.parse_args()
    
    calib_path = Path(args.calibration)
    meta_path = Path(args.meta)
    
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration not found: {calib_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta not found: {meta_path}")
    
    calib = load_json(calib_path)
    meta = load_json(meta_path)
    m = meta["mapping"] if "mapping" in meta else meta
    
    # Extract calibration points
    points = calib["calibration_points"]
    
    if len(points) < 3:
        raise ValueError("Need at least 3 calibration points for meaningful analysis")
    
    print("=" * 80)
    print("CALIBRATION ANALYSIS")
    print("=" * 80)
    print(f"\nImage size: {calib['image_dimensions']['width']}x{calib['image_dimensions']['height']}")
    print(f"Calibration points: {len(points)}")
    print(f"Source bounds: W={m['bounds_west']:.6f}, S={m['bounds_south']:.6f}, E={m['bounds_east']:.6f}, N={m['bounds_north']:.6f}")
    print(f"Rotation: {m['rot_deg']}°")
    print()
    
    # Build arrays
    geo_coords = np.array([[p["geographic"]["lon"], p["geographic"]["lat"]] for p in points])
    clicked_pixels = np.array([[p["pixel"]["x"], p["pixel"]["y"]] for p in points])
    
    # Test current transform
    print("=" * 80)
    print("TESTING CURRENT TRANSFORM (from bounds_rotcrop.json)")
    print("=" * 80)
    
    predicted_pixels = []
    errors = []
    
    for i, (point, geo, clicked) in enumerate(zip(points, geo_coords, clicked_pixels)):
        pred_x, pred_y = lonlat_to_outpx_current(m, geo[0], geo[1])
        predicted_pixels.append([pred_x, pred_y])
        
        err_x = clicked[0] - pred_x
        err_y = clicked[1] - pred_y
        err_dist = math.sqrt(err_x**2 + err_y**2)
        errors.append(err_dist)
        
        print(f"\n{i+1}. {point['name']}")
        print(f"   Geographic: ({geo[0]:.6f}°, {geo[1]:.6f}°)")
        print(f"   You clicked:     ({clicked[0]:7.1f}, {clicked[1]:7.1f}) px")
        print(f"   Current predicts: ({pred_x:7.1f}, {pred_y:7.1f}) px")
        print(f"   Error: ({err_x:+7.1f}, {err_y:+7.1f}) px  |  Distance: {err_dist:.1f} px")
    
    predicted_pixels = np.array(predicted_pixels)
    errors = np.array(errors)
    
    print("\n" + "=" * 80)
    print("ERROR STATISTICS")
    print("=" * 80)
    print(f"Mean error:   {errors.mean():.1f} px")
    print(f"Max error:    {errors.max():.1f} px")
    print(f"Std dev:      {errors.std():.1f} px")
    
    # Fit optimal affine transform directly from geo->pixel
    print("\n" + "=" * 80)
    print("FITTING OPTIMAL AFFINE TRANSFORM (geo -> pixel)")
    print("=" * 80)
    
    affine = fit_affine_transform(geo_coords, clicked_pixels)
    residuals, distances = compute_residuals(geo_coords, clicked_pixels, affine)
    
    print("\nAffine matrix (lon,lat -> x,y):")
    print(f"  x = {affine[0,0]:11.6f} * lon + {affine[0,1]:11.6f} * lat + {affine[0,2]:11.6f}")
    print(f"  y = {affine[1,0]:11.6f} * lon + {affine[1,1]:11.6f} * lat + {affine[1,2]:11.6f}")
    
    print("\nFit quality:")
    print(f"  Mean residual:   {distances.mean():.3f} px")
    print(f"  Max residual:    {distances.max():.3f} px")
    print(f"  RMS residual:    {np.sqrt((distances**2).mean()):.3f} px")
    
    if args.verbose:
        print("\nPer-point residuals:")
        for i, (point, dist, res) in enumerate(zip(points, distances, residuals)):
            print(f"  {i+1}. {point['name']:20s}: {dist:6.2f} px  ({res[0]:+6.2f}, {res[1]:+6.2f})")
    
    # Diagnose the problem
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    if errors.mean() < 5:
        print("✓ Current transform is ACCURATE (mean error < 5 px)")
        print("  Your rotation/crop/squish math is correct!")
    elif errors.mean() < 20:
        print("⚠ Current transform has SMALL SYSTEMATIC ERROR (mean error < 20 px)")
        print("  Likely cause: rotation center offset or tx/ty computation issue")
        print("  Recommendation: Use the fitted affine transform to correct")
    else:
        print("✗ Current transform is BROKEN (mean error > 20 px)")
        print("  Likely cause: rotation math is fundamentally wrong")
        print("  Recommendation: Rebuild transform from scratch using fitted affine")
    
    # Check for systematic bias
    mean_err_x = (clicked_pixels[:, 0] - predicted_pixels[:, 0]).mean()
    mean_err_y = (clicked_pixels[:, 1] - predicted_pixels[:, 1]).mean()
    
    print(f"\nSystematic bias: ({mean_err_x:+.1f}, {mean_err_y:+.1f}) px")
    if abs(mean_err_x) > 10 or abs(mean_err_y) > 10:
        print("  → Large bias suggests a translation offset in the rotation transform")
    
    # Save corrected affine
    correction = {
        "fitted_affine": {
            "matrix": affine.tolist(),
            "description": "Direct lon,lat -> x,y affine transform fitted to calibration points"
        },
        "current_transform_errors": {
            "mean_px": float(errors.mean()),
            "max_px": float(errors.max()),
            "std_px": float(errors.std())
        },
        "fitted_transform_quality": {
            "mean_residual_px": float(distances.mean()),
            "max_residual_px": float(distances.max()),
            "rms_residual_px": float(np.sqrt((distances**2).mean()))
        },
        "calibration_source": str(calib_path),
        "original_meta": str(meta_path)
    }
    
    out_path = calib_path.parent / "transform_correction.json"
    out_path.write_text(json.dumps(correction, indent=2))
    
    print(f"\n✓ Saved correction to: {out_path}")
    print("\nNext steps:")
    print("  1. If mean error > 10 px, your transform needs fixing")
    print("  2. Run the correction script to rebuild bounds_rotcrop.json")
    print("  3. Re-run bake_elev_rg.py with corrected transform")


if __name__ == "__main__":
    main()