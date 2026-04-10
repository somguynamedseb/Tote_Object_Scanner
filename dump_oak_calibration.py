#!/usr/bin/env python3
"""
Dump OAK-D SR Factory Calibration
==================================
Pulls intrinsics, distortion, and extrinsics for CAM_B (left) and CAM_C
(right) from the device EEPROM and prints them in a form you can paste
directly into another script.

Usage:
    python dump_oak_calibration.py
    python dump_oak_calibration.py --width 1280 --height 800
    python dump_oak_calibration.py --json calib.json

The resolution matters: DepthAI scales the intrinsics to the resolution
you request, so pass the same width/height you use for capture.
"""

import argparse
import json
import sys

import depthai as dai
import numpy as np


def fmt_matrix(name: str, mat) -> str:
    arr = np.asarray(mat)
    rows = [", ".join(f"{v: .6f}" for v in row) for row in arr]
    inner = ",\n    ".join(f"[{r}]" for r in rows)
    return f"{name} = np.array([\n    {inner},\n])"


def fmt_vector(name: str, vec) -> str:
    arr = np.asarray(vec).flatten()
    vals = ", ".join(f"{v: .6f}" for v in arr)
    return f"{name} = np.array([{vals}])"


def main():
    parser = argparse.ArgumentParser(description="Dump OAK-D SR calibration")
    parser.add_argument("--width", type=int, default=1280,
                        help="Capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=800,
                        help="Capture height (default: 800)")
    parser.add_argument("--json", default=None,
                        help="Optional path to also write JSON output")
    args = parser.parse_args()

    try:
        with dai.Device() as device:
            calib = device.readCalibration()
    except Exception as e:
        print(f"Failed to read calibration from device: {e}", file=sys.stderr)
        sys.exit(1)

    left_socket = dai.CameraBoardSocket.CAM_B
    right_socket = dai.CameraBoardSocket.CAM_C

    K_left = calib.getCameraIntrinsics(left_socket, args.width, args.height)
    D_left = calib.getDistortionCoefficients(left_socket)

    K_right = calib.getCameraIntrinsics(right_socket, args.width, args.height)
    D_right = calib.getDistortionCoefficients(right_socket)

    # Extrinsics: transform from right camera frame to left camera frame.
    # This is what stereoRectify expects as (R, T).
    extrinsics = calib.getCameraExtrinsics(left_socket, right_socket)
    extrinsics = np.asarray(extrinsics)
    R = extrinsics[:3, :3]
    T = extrinsics[:3, 3]  # translation in cm per DepthAI convention

    # DepthAI reports translation in centimeters — convert to millimeters
    # so everything downstream is in mm.
    T_mm = T * 10.0
    baseline_mm = float(np.linalg.norm(T_mm))

    print("# " + "=" * 70)
    print(f"# OAK-D SR calibration @ {args.width}x{args.height}")
    print(f"# Baseline (CAM_B → CAM_C): {baseline_mm:.3f} mm")
    print("# " + "=" * 70)
    print()
    print("import numpy as np")
    print()
    print(f"IMAGE_WIDTH  = {args.width}")
    print(f"IMAGE_HEIGHT = {args.height}")
    print(f"BASELINE_MM  = {baseline_mm:.6f}")
    print()
    print(fmt_matrix("K_LEFT", K_left))
    print()
    print(fmt_vector("D_LEFT", D_left))
    print()
    print(fmt_matrix("K_RIGHT", K_right))
    print()
    print(fmt_vector("D_RIGHT", D_right))
    print()
    print("# Rotation: right camera frame → left camera frame")
    print(fmt_matrix("R_RIGHT_TO_LEFT", R))
    print()
    print("# Translation (mm): right camera frame → left camera frame")
    print(fmt_vector("T_RIGHT_TO_LEFT_MM", T_mm))

    if args.json:
        payload = {
            "width": args.width,
            "height": args.height,
            "baseline_mm": baseline_mm,
            "K_left": np.asarray(K_left).tolist(),
            "D_left": np.asarray(D_left).tolist(),
            "K_right": np.asarray(K_right).tolist(),
            "D_right": np.asarray(D_right).tolist(),
            "R_right_to_left": R.tolist(),
            "T_right_to_left_mm": T_mm.tolist(),
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n# JSON written to {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
