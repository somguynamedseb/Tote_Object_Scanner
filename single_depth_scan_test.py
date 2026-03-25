#!/usr/bin/env python3
"""
G-code + OAK-D Depth Map Grid Scanner
=======================================
1. Homes the machine with $H and waits for user confirmation.
2. Reads grid positions and settings from a YAML config file.
3. Moves through the grid capturing depth maps as separate files.

Requirements:
    pip install depthai opencv-python numpy pyserial pyyaml

Usage:
    python gcode_depth_scanner.py                        # uses scan_config.yaml
    python gcode_depth_scanner.py -c my_config.yaml      # custom config
    python gcode_depth_scanner.py --preview               # camera preview only
"""

import argparse
from datetime import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import depthai as dai
import numpy as np
import serial
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

def load_config(path="scan_config.yaml") -> dict:
    """Load and return the YAML config, with defaults for missing keys."""
    p = Path(path)

    with open(p) as f:
        cfg = yaml.safe_load(f)

    # Apply defaults for anything missing
    cfg.setdefault("serial", {})
    cfg["serial"].setdefault("port", "/dev/ttyUSB0")
    cfg["serial"].setdefault("baudrate", 115200)
    cfg["serial"].setdefault("timeout", 10.0)

    cfg.setdefault("camera", {})
    cfg["camera"].setdefault("subpixel", True)
    cfg["camera"].setdefault("lr_check", True)
    cfg["camera"].setdefault("extended_disparity", True)

    cfg.setdefault("motion", {})
    cfg["motion"].setdefault("feedrate", 1000)
    cfg["motion"].setdefault("settle_time", 0.5)

    cfg.setdefault("grid", {})
    cfg["grid"].setdefault("x_start", 0.0)
    cfg["grid"].setdefault("x_stop", 100.0)
    cfg["grid"].setdefault("x_step", 50.0)
    cfg["grid"].setdefault("y_start", 0.0)
    cfg["grid"].setdefault("y_stop", 100.0)
    cfg["grid"].setdefault("y_step", 50.0)

    cfg.setdefault("output", {})
    cfg["output"].setdefault("directory", "scan_output")
    cfg["output"].setdefault("save_raw_npy", True)
    cfg["output"].setdefault("save_depth_png", True)

    return cfg


# ---------------------------------------------------------------------------
# OAK-D Depth Camera
# ---------------------------------------------------------------------------

class OakDDepthCamera:
    """Manages the OAK-D stereo depth pipeline via the new DepthAI API."""

    def __init__(self, cam_cfg: dict):
        self.cam_cfg = cam_cfg
        self._pipeline: Optional[dai.Pipeline] = None
        self._depth_queue = None
        self._disparity_queue = None
        self._max_disparity = 1

        # Custom colormap: JET but with zero-disparity pixels black
        self._colormap = cv2.applyColorMap(
            np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET,
        )
        self._colormap[0] = [0, 0, 0]

    def _build_pipeline(self) -> dai.Pipeline:
        pipeline = dai.Pipeline()

        # Left mono camera
        mono_left = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_B,
        )
        # Right mono camera
        mono_right = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_C,
        )

        # Stereo depth
        stereo = pipeline.create(dai.node.StereoDepth)

        # Link full-resolution mono outputs into stereo
        mono_left.requestFullResolutionOutput().link(stereo.left)
        mono_right.requestFullResolutionOutput().link(stereo.right)

        stereo.setRectification(True)
        stereo.setExtendedDisparity(True)
        stereo.setLeftRightCheck(True)

        # Output queues created directly from node outputs
        self._depth_queue = stereo.depth.createOutputQueue()
        self._disparity_queue = stereo.disparity.createOutputQueue()

        return pipeline

    def start(self):
        log.info("Starting OAK-D depth pipeline …")
        self._pipeline = self._build_pipeline()
        self._pipeline.start()
        log.info("OAK-D pipeline running.")

    def stop(self):
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
            log.info("OAK-D pipeline stopped.")

    def get_depth_frame(self) -> Optional[np.ndarray]:
        """Return the latest depth frame as a numpy array."""
        if self._depth_queue is None:
            return None
        msg = self._depth_queue.get()
        assert isinstance(msg, dai.ImgFrame)
        return msg.getFrame()

    def get_disparity_frame(self) -> Optional[np.ndarray]:
        """Return the latest disparity frame as a numpy array."""
        if self._disparity_queue is None:
            return None
        msg = self._disparity_queue.get()
        assert isinstance(msg, dai.ImgFrame)
        return msg.getFrame()

    def disparity_to_colormap(self, disparity_frame: np.ndarray) -> np.ndarray:
        """Convert a disparity frame to an 8-bit colourmap image."""
        self._max_disparity = max(self._max_disparity, np.max(disparity_frame))
        normalized = ((disparity_frame / self._max_disparity) * 255).astype(np.uint8)
        return cv2.applyColorMap(normalized, self._colormap)

    def flush_queues(self):
        """Drain any buffered frames so the next get() returns a fresh capture."""
        for q in (self._depth_queue, self._disparity_queue):
            if q is None:
                continue
            while q.has():
                q.get()

    def capture(self, warmup_frames=30):
        """Capture a depth + disparity pair after letting auto-exposure stabilize."""
        log.info("Warming up camera (%d frames) …", warmup_frames)
        for i in range(warmup_frames):
            self._depth_queue.get()
            self._disparity_queue.get()
        log.info("Warm-up complete, capturing frame.")

        depth = self.get_depth_frame()
        disparity = self.get_disparity_frame()
        return depth, disparity


# ---------------------------------------------------------------------------
# Main Scan Routine
# ---------------------------------------------------------------------------

cfg = load_config()
camera = OakDDepthCamera(cfg["camera"])

# ── 1. Connect hardware ──
camera.start()

# ── 2. Capture with warm-up ──
depth, disparity = camera.capture(warmup_frames=30)

# ── 3. Save raw depth (millimeters) for data use ──
raw_path = "test_data.npy"
np.save(str(raw_path), depth)
log.info("Saved raw depth → %s", raw_path)

# ── 4. Colorize the DISPARITY frame (matches reference script) ──
cmap = camera.disparity_to_colormap(disparity)
png_path = "test_data_img.png"
cv2.imwrite(str(png_path), cmap)
log.info("Saved colorized disparity → %s", png_path)

# ── 5. Shut down ──
camera.stop()