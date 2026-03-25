#!/usr/bin/env python3
"""
View a single depth map as a 3D point cloud.

Usage:
    python depth_viewer.py path/to/depth.npy
    python depth_viewer.py path/to/depth.npy --min 200 --max 5000
    python depth_viewer.py path/to/depth.npy --fx 450 --fy 450 --cx 320 --cy 200

If an OAK-D is connected, intrinsics are read from the device automatically
and scaled to match the depth map resolution.
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def get_oakd_intrinsics(width: int, height: int) -> tuple[float, float, float, float]:
    """
    Read intrinsics from a connected OAK-D, scaled to the given resolution.

    Depth is aligned to the LEFT camera (CAM_B) by default in StereoDepth,
    so we read intrinsics from CAM_B and ask the calibration to rescale them
    to match the actual depth map dimensions.
    """
    try:
        import depthai as dai
        with dai.Device() as device:
            calib = device.readCalibration()
            # Use CAM_B (left) — default stereo depth alignment target
            # Pass the depth map resolution so intrinsics are scaled correctly
            intrinsics = calib.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_B, width, height,
            )
            fx = intrinsics[0][0]
            fy = intrinsics[1][1]
            cx = intrinsics[0][2]
            cy = intrinsics[1][2]
            print(f"OAK-D intrinsics (for {width}x{height}): "
                  f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
            return fx, fy, cx, cy
    except Exception as e:
        print(f"Could not read OAK-D calibration ({e}). Using defaults.")
        # Rough defaults for 640x400 mono — scale if your map differs
        default_fx = 450.0 * (width / 640.0)
        default_fy = 450.0 * (height / 400.0)
        default_cx = width / 2.0
        default_cy = height / 2.0
        print(f"Scaled defaults (for {width}x{height}): "
              f"fx={default_fx:.1f} fy={default_fy:.1f} "
              f"cx={default_cx:.1f} cy={default_cy:.1f}")
        return default_fx, default_fy, default_cx, default_cy


def depth_to_cloud(
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    min_depth: float=0, 
    max_depth: float=10000,
    max_radius: float = 0,
) -> o3d.geometry.PointCloud:
    """Back-project a depth frame into a coloured 3D point cloud.

    Parameters
    ----------
    max_radius : float
        Maximum lateral (XY) distance from the optical axis in mm.
        Points further from center are discarded. 0 = no limit.
    """
    h, w = depth.shape[:2]

    u = np.arange(w, dtype=np.float64)
    v = np.arange(h, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)

    z = depth.astype(np.float64)
    valid = (z > min_depth) & (z < max_depth)

    z_valid = z[valid]
    x = (uu[valid] - cx) * z_valid / fx
    # Flip Y so the cloud appears right-side-up in the viewer
    y = -(vv[valid] - cy) * z_valid / fy

    # --- Lateral radius filter: keep only points within a cylinder ---
    if max_radius > 0:
        lateral_dist = np.sqrt(x**2 + y**2)
        within = lateral_dist <= max_radius
        x = x[within]
        y = y[within]
        z_valid = z_valid[within]
        print(f"Radius filter: kept {within.sum()} / {len(within)} points "
              f"(radius <= {max_radius:.0f} mm)")

    points = np.column_stack((x, y, z_valid))

    if len(points) == 0:
        cloud = o3d.geometry.PointCloud()
        return cloud

    # Colour by depth using JET colormap
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


def plot_distributions(
    raw_pts: np.ndarray,
    filtered_pts: np.ndarray,
    output_path: str = "depth_distributions.png",
    bins: int = 100,
):
    """
    Plot side-by-side histograms of point distributions along Z (depth),
    X (lateral), and Y (vertical) for raw vs filtered point clouds.

    Parameters
    ----------
    raw_pts : (N, 3) array — unfiltered points (X, Y, Z).
    filtered_pts : (M, 3) array — filtered points (X, Y, Z).
    output_path : where to save the figure.
    bins : histogram bin count.
    """
    axes_labels = ["X (lateral, mm)", "Y (vertical, mm)", "Z (depth, mm)"]
    axis_indices = [0, 1, 2]

    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Point Distribution — Unfiltered vs Filtered", fontsize=14, y=0.98)

    for row, (ax_idx, label) in enumerate(zip(axis_indices, axes_labels)):
        raw_vals = raw_pts[:, ax_idx]
        filt_vals = filtered_pts[:, ax_idx]


        raw_lo, raw_hi = np.percentile(raw_vals, [0.5, 99.5])
        raw_edges = np.linspace(raw_lo, raw_hi, bins + 1)
        
        flt_lo, flt_hi = np.percentile(filt_vals, [0.5, 99.5])
        flt_edges = np.linspace(flt_lo, flt_hi, bins + 1)

        # Unfiltered (left column)
        ax_raw = axs[row, 0]
        ax_raw.hist(raw_vals, bins=raw_edges, color="#4a90d9", edgecolor="none", alpha=0.85)
        ax_raw.set_xlabel(label)
        ax_raw.set_ylabel("Point count")
        if row == 0:
            ax_raw.set_title(f"Unfiltered  ({len(raw_pts):,} pts)")
        ax_raw.axvline(raw_vals.mean(), color="red", ls="--", lw=1, label=f"mean {raw_vals.mean():.0f}")
        ax_raw.legend(fontsize=8)

        # Filtered (right column)
        ax_filt = axs[row, 1]
        ax_filt.hist(filt_vals, bins=flt_edges, color="#e07b39", edgecolor="none", alpha=0.85)
        ax_filt.set_xlabel(label)
        ax_filt.set_ylabel("Point count")
        if row == 0:
            ax_filt.set_title(f"Filtered  ({len(filtered_pts):,} pts)")
        ax_filt.axvline(filt_vals.mean(), color="red", ls="--", lw=1, label=f"mean {filt_vals.mean():.0f}")
        ax_filt.legend(fontsize=8)

        # Match Y-axis scale so you can see how much was removed
        y_max = max(ax_raw.get_ylim()[1], ax_filt.get_ylim()[1])
        ax_raw.set_ylim(0, y_max)
        ax_filt.set_ylim(0, y_max)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nDistribution plot saved → {output_path}")


def main(min_depth_def = 250,max_depth_def = 450,max_rad_def = 300):
    parser = argparse.ArgumentParser(
        description="View a single depth file as a 3D point cloud",
    )
    parser.add_argument("file", help="Path to a .npy depth file")
    parser.add_argument("--min", type=float, default=min_depth_def,
                        help="Min depth in mm (default: 100)")
    parser.add_argument("--max", type=float, default=max_depth_def,
                        help="Max depth in mm (default: 10000)")
    parser.add_argument("--fx", type=float, default=None,
                        help="Focal length X (auto from OAK-D if omitted)")
    parser.add_argument("--fy", type=float, default=None,
                        help="Focal length Y (auto from OAK-D if omitted)")
    parser.add_argument("--cx", type=float, default=None,
                        help="Principal point X (auto from OAK-D if omitted)")
    parser.add_argument("--cy", type=float, default=None,
                        help="Principal point Y (auto from OAK-D if omitted)")
    parser.add_argument("--radius", type=float, default=max_rad_def,
                        help="Max lateral (XY) distance from optical axis in mm. "
                             "Crops the pyramid to a cylinder. 0 = no limit.")
    parser.add_argument("--voxel", type=float, default=0,
                        help="Voxel downsample size in mm (0 = off)")
    parser.add_argument("--sor", type=int, default=0,
                        help="Statistical outlier removal: number of neighbors "
                             "to consider (e.g. 20). 0 = off.")
    parser.add_argument("--sor-std", type=float, default=2.0,
                        help="SOR std deviation threshold (default: 2.0). "
                             "Lower = more aggressive filtering.")
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
        depth = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
                 if img.ndim == 3 else img.astype(np.float64))
    else:
        print(f"Unsupported file type: {path.suffix}")
        sys.exit(1)

    h, w = depth.shape[:2]

    # --- Raw data stats (before any filtering) ---
    valid_pixels = depth[depth > 0]
    print(f"\n{'='*50}")
    print(f"  RAW DATA")
    print(f"{'='*50}")
    print(f"File:      {path.name}")
    print(f"Shape:     {depth.shape} ({w}x{h})")
    if valid_pixels.size:
        print(f"Pixels:    {valid_pixels.size}")
        print(f"Min (>0):  {valid_pixels.min():.1f} mm")
        print(f"Max:       {depth.max():.1f} mm")
        print(f"Mean (>0): {valid_pixels.mean():.1f} mm")
        print(f"Std (>0):  {valid_pixels.std():.1f} mm")
    else:
        print("No valid (>0) pixels found — depth map may be empty.")
        sys.exit(1)

    # Intrinsics — use manual overrides if ALL four are given, else auto-detect
    if all(v is not None for v in (args.fx, args.fy, args.cx, args.cy)):
        fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
        print(f"Manual intrinsics: fx={fx:.1f} fy={fy:.1f} "
              f"cx={cx:.1f} cy={cy:.1f}")
    else:
        fx, fy, cx, cy = get_oakd_intrinsics(w, h)

    # Build unfiltered cloud (depth range only, no radius) for comparison
    unfiltered_cloud = depth_to_cloud(depth, fx, fy, cx, cy)
    raw_pts = np.asarray(unfiltered_cloud.points).copy()

    # Build filtered cloud
    cloud = depth_to_cloud(depth, fx, fy, cx, cy, args.min, args.max,
                           max_radius=args.radius)

    def print_cloud_stats(cloud, label):
        """Print depth stats from the Z values of a point cloud."""
        pts = np.asarray(cloud.points)
        if len(pts) == 0:
            print(f"  Points: 0")
            return
        z = pts[:, 2]  # Z = depth
        print(f"\n{'='*50}")
        print(f"  {label}")
        print(f"{'='*50}")
        print(f"Points:    {len(pts)}")
        print(f"Min depth: {z.min():.1f} mm")
        print(f"Max depth: {z.max():.1f} mm")
        print(f"Mean:      {z.mean():.1f} mm")
        print(f"Std:       {z.std():.1f} mm")

    filters_applied = ["depth range"]
    if args.radius > 0:
        filters_applied.append(f"radius <= {args.radius:.0f} mm")
    print_cloud_stats(cloud, f"AFTER {' + '.join(filters_applied)}")

    if len(cloud.points) == 0:
        print("No points within depth/radius range — try adjusting --min / --max / --radius.")
        sys.exit(1)

    if args.voxel > 0:
        cloud = cloud.voxel_down_sample(voxel_size=args.voxel)
        print_cloud_stats(cloud, f"AFTER voxel downsample ({args.voxel:.1f} mm)")

    if args.sor > 0:
        cloud, inlier_idx = cloud.remove_statistical_outlier(
            nb_neighbors=args.sor, std_ratio=args.sor_std,
        )
        print_cloud_stats(cloud, f"AFTER SOR (neighbors={args.sor}, std={args.sor_std})")

    # --- Distribution comparison plots ---
    filtered_pts = np.asarray(cloud.points)
    plot_distributions(raw_pts, filtered_pts,
                       output_path=str(path.with_suffix("")) + "_distributions.png")

    # Show
    print("\n3D viewer open — close the window to exit.")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Depth: {path.name}", width=1280, height=720)
    vis.add_geometry(cloud)

    opt = vis.get_render_option()
    opt.point_size = 1.5
    opt.background_color = np.array([0.1, 0.1, 0.1])

    # Set a front-facing viewpoint so the scene isn't viewed from an odd angle
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_lookat(cloud.get_center())
    ctr.set_zoom(0.5)

    vis.run()
    vis.destroy_window()

    


if __name__ == "__main__":
    main()