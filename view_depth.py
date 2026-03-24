#!/usr/bin/env python3
"""
View a single depth map as a 3D point cloud.

Usage:
    python view_depth.py path/to/depth_0001_X0.0_Y0.0.npy
    python view_depth.py path/to/depth_0001_X0.0_Y0.0.npy --min 200 --max 5000
    python view_depth.py path/to/depth_0001_X0.0_Y0.0.npy --fx 450 --fy 450 --cx 320 --cy 200

If an OAK-D is connected, intrinsics are read from the device automatically.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def get_oakd_intrinsics() -> tuple[float, float, float, float]:
    """Read intrinsics from a connected OAK-D, or return defaults."""
    try:
        import depthai as dai
        with dai.Device() as device:
            calib = device.readCalibration()
            intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C)
            fx = intrinsics[0][0]
            fy = intrinsics[1][1]
            cx = intrinsics[0][2]
            cy = intrinsics[1][2]
            print(f"OAK-D intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
            return fx, fy, cx, cy
    except Exception as e:
        print(f"Could not read OAK-D calibration ({e}). Using defaults.")
        return 450.0, 450.0, 320.0, 200.0


def depth_to_cloud(
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    min_depth: float, max_depth: float,
) -> o3d.geometry.PointCloud:
    """Back-project a depth frame into a coloured 3D point cloud."""
    h, w = depth.shape[:2]

    u = np.arange(w, dtype=np.float64)
    v = np.arange(h, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)

    z = depth.astype(np.float64)
    valid = (z > min_depth) & (z < max_depth)

    z_valid = z[valid]
    x = (uu[valid] - cx) * z_valid / fx
    y = (vv[valid] - cy) * z_valid / fy

    points = np.column_stack((x, y, z_valid))

    # Colour by depth
    z_min, z_max = z_valid.min(), z_valid.max()
    z_norm = (z_valid - z_min) / (z_max - z_min + 1e-6)
    colors_bgr = cv2.applyColorMap(
        (z_norm * 255).astype(np.uint8).reshape(-1, 1),
        cv2.COLORMAP_JET,
    ).reshape(-1, 3)
    colors_rgb = colors_bgr[:, ::-1] / 255.0

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors_rgb)
    return cloud


def main():
    parser = argparse.ArgumentParser(description="View a single depth file as a 3D point cloud")
    parser.add_argument("file", help="Path to a .npy depth file")
    parser.add_argument("--min", type=float, default=100, help="Min depth in mm (default: 100)")
    parser.add_argument("--max", type=float, default=10000, help="Max depth in mm (default: 10000)")
    parser.add_argument("--fx", type=float, default=None, help="Focal length X (auto from OAK-D if omitted)")
    parser.add_argument("--fy", type=float, default=None, help="Focal length Y (auto from OAK-D if omitted)")
    parser.add_argument("--cx", type=float, default=None, help="Principal point X (auto from OAK-D if omitted)")
    parser.add_argument("--cy", type=float, default=None, help="Principal point Y (auto from OAK-D if omitted)")
    parser.add_argument("--voxel", type=float, default=0, help="Voxel downsample size in mm (0 = off)")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    # Load depth
    if path.suffix == ".npy":
        depth = np.load(str(path))
    elif path.suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Could not read image: {path}")
            sys.exit(1)
        depth = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) if img.ndim == 3 else img.astype(np.float64)
    else:
        print(f"Unsupported file type: {path.suffix}")
        sys.exit(1)

    # Stats
    valid = depth[depth > 0]
    print(f"File:      {path.name}")
    print(f"Shape:     {depth.shape}")
    print(f"Min (>0):  {valid.min():.1f} mm" if valid.size else "Min: no valid pixels")
    print(f"Max:       {depth.max():.1f} mm")
    print(f"Mean (>0): {valid.mean():.1f} mm" if valid.size else "Mean: no valid pixels")

    # Intrinsics
    if args.fx is not None and args.fy is not None and args.cx is not None and args.cy is not None:
        fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
    else:
        fx, fy, cx, cy = get_oakd_intrinsics()

    # Build cloud
    cloud = depth_to_cloud(depth, fx, fy, cx, cy, args.min, args.max)
    print(f"Points:    {len(cloud.points)}")

    if args.voxel > 0:
        cloud = cloud.voxel_down_sample(voxel_size=args.voxel)
        print(f"After downsample: {len(cloud.points)}")

    # Show
    print("\n3D viewer open — close the window to exit.")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Depth: {path.name}", width=1280, height=720)
    vis.add_geometry(cloud)

    opt = vis.get_render_option()
    opt.point_size = 1.5
    opt.background_color = np.array([0.1, 0.1, 0.1])

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()