#!/usr/bin/env python3
"""
View Point Cloud
================
Loads an Nx3 point cloud from a .npy file produced by build_pointcloud.py
and displays it interactively with Open3D.

Controls:
    Left-drag       rotate
    Right-drag      pan
    Scroll          zoom
    R               reset view
    +/-             change point size
    H               print full Open3D help
    Q / Esc         quit

Point colors encode Z (height), using the Turbo colormap.

Usage:
    python view_pointcloud.py pointcloud.npy
    python view_pointcloud.py pointcloud.npy --point-size 2.5
    python view_pointcloud.py pointcloud.npy --color-axis x
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("open3d is required — install with: pip install open3d", file=sys.stderr)
    sys.exit(1)

try:
    import matplotlib.cm as cm
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def colorize(values: np.ndarray) -> np.ndarray:
    """Map a 1D array of values to RGB in [0,1] using Turbo (or viridis fallback)."""
    lo, hi = np.percentile(values, [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((values - lo) / (hi - lo), 0.0, 1.0)

    if _HAS_MPL:
        try:
            cmap = cm.get_cmap("turbo")
        except ValueError:
            cmap = cm.get_cmap("viridis")
        colors = cmap(norm)[:, :3]
    else:
        # Simple blue→red ramp as a fallback
        colors = np.zeros((norm.size, 3), dtype=np.float64)
        colors[:, 0] = norm
        colors[:, 2] = 1.0 - norm
    return colors.astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description="View combined stereo point cloud")
    parser.add_argument("npy", help="Path to .npy point cloud (Nx3)")
    parser.add_argument("--point-size", type=float, default=2.0,
                        help="Render point size (default: 2.0)")
    parser.add_argument("--color-axis", choices=["x", "y", "z"], default="z",
                        help="Axis used for coloring (default: z)")
    parser.add_argument("--voxel-downsample", type=float, default=0.0,
                        help="Voxel size in mm for downsampling (0 = off)")
    parser.add_argument("--no-axes", action="store_true",
                        help="Hide the coordinate axes gizmo")
    args = parser.parse_args()

    path = Path(args.npy)
    if not path.exists():
        log.error("File not found: %s", path)
        sys.exit(1)

    pts = np.load(path)
    if pts.ndim != 2 or pts.shape[1] != 3:
        log.error("Expected Nx3 array, got shape %s", pts.shape)
        sys.exit(1)

    log.info("Loaded %d points from %s", pts.shape[0], path)
    log.info("  X: %.1f .. %.1f mm", pts[:, 0].min(), pts[:, 0].max())
    log.info("  Y: %.1f .. %.1f mm", pts[:, 1].min(), pts[:, 1].max())
    log.info("  Z: %.1f .. %.1f mm", pts[:, 2].min(), pts[:, 2].max())

    axis_idx = {"x": 0, "y": 1, "z": 2}[args.color_axis]
    colors = colorize(pts[:, axis_idx])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if args.voxel_downsample > 0:
        before = len(pcd.points)
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_downsample)
        log.info("Voxel-downsampled %d → %d points (voxel=%.2f mm)",
                 before, len(pcd.points), args.voxel_downsample)

    geometries = [pcd]
    if not args.no_axes:
        # Size the axes relative to the cloud
        extent = pts.max(axis=0) - pts.min(axis=0)
        axis_size = float(max(extent.max() * 0.1, 10.0))
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=axis_size,
            origin=pts.min(axis=0).astype(np.float64),
        )
        geometries.append(frame)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"Point Cloud — {path.name}",
        width=1280,
        height=800,
    )
    for g in geometries:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.array([0.05, 0.05, 0.08])

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
