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


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load and return the YAML config, with defaults for missing keys."""
    p = Path(path)
    if not p.exists():
        log.error("Config file not found: %s", path)
        sys.exit(1)

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

        # Left mono camera — .build() binds the socket, no setResolution/setCamera needed
        mono_left = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_B,
        )

        # Right mono camera
        mono_right = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_C,
        )

        # Stereo depth
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.ACCURACY)

        # Link full-resolution mono outputs into stereo
        # FPS is controlled here via the fps parameter
        mono_left.requestFullResolutionOutput(fps=self.cam_cfg["fps"]).link(stereo.left)
        mono_right.requestFullResolutionOutput(fps=self.cam_cfg["fps"]).link(stereo.right)

        # v3 uses setters directly on the stereo node, not on initialConfig
        stereo.setRectification(True)
        stereo.setExtendedDisparity(self.cam_cfg["extended_disparity"])
        stereo.setLeftRightCheck(self.cam_cfg["lr_check"])
        stereo.setSubpixel(self.cam_cfg["subpixel"])

        # These remain on initialConfig
        stereo.initialConfig.setSubpixelFractionalBits(self.cam_cfg["subpixel_frac_bits"])
        stereo.initialConfig.setDisparityShift(self.cam_cfg["disparity_shift"])
        stereo.initialConfig.setConfidenceThreshold(self.cam_cfg["confidence"])
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

        stereo.setPostProcessingHardwareResources(3, 3)  # (numShaves, numMemorySlices)

        # Speckle filter
        stereo.initialConfig.postProcessing.speckleFilter.enable = True
        stereo.initialConfig.postProcessing.speckleFilter.speckleRange = 50

        # Temporal filter
        stereo.initialConfig.postProcessing.temporalFilter.enable = True
        stereo.initialConfig.postProcessing.temporalFilter.alpha = 0.15
        stereo.initialConfig.postProcessing.temporalFilter.delta = 0

        # Spatial filter — disabled for accuracy
        stereo.initialConfig.postProcessing.spatialFilter.enable = False

        # Threshold filter
        stereo.initialConfig.postProcessing.thresholdFilter.minRange = 100
        stereo.initialConfig.postProcessing.thresholdFilter.maxRange = 1000

        # Decimation
        stereo.initialConfig.postProcessing.decimationFilter.decimationFactor = 2
        stereo.initialConfig.postProcessing.decimationFilter.decimationMode = (
            dai.StereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.NON_ZERO_MEDIAN
        )

        # Bilateral
        stereo.initialConfig.setBilateralFilterSigma(0)

        # Brightness filter
        stereo.initialConfig.postProcessing.brightnessFilter.minBrightness = 1
        stereo.initialConfig.postProcessing.brightnessFilter.maxBrightness = 254

        # v3: output queues directly from node outputs, no XLinkOut needed
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

    def depth_to_colormap(self, depth_frame: np.ndarray) -> np.ndarray:
        """Convert a depth/disparity frame to an 8-bit colourmap image."""
        self._max_disparity = max(self._max_disparity, np.max(depth_frame))
        normalized = ((depth_frame / self._max_disparity) * 255).astype(np.uint8)
        return cv2.applyColorMap(normalized, self._colormap)

    def flush_queues(self):
        """Drain any buffered frames so the next get() returns a fresh capture."""
        for q in (self._depth_queue, self._disparity_queue):
            if q is None:
                continue
            while q.has():
                q.get()
                
    def capture(self):
        """Capture a depth + disparity pair after letting auto-exposure stabilize."""
        warmup_frames = self.cam_cfg["warmup_frames"]
        log.info("Warming up camera (%d frames) …", warmup_frames)
        for i in range(warmup_frames):
            self._depth_queue.get()
            self._disparity_queue.get()
        log.info("Warm-up complete, capturing frame.")

        depth = self.get_depth_frame()
        disparity = self.get_disparity_frame()
        return depth, disparity


# ---------------------------------------------------------------------------
# G-code Serial Sender
# ---------------------------------------------------------------------------

class GcodeSender:
    """Send G-code / GRBL commands over serial and wait for acknowledgement."""

    def __init__(self, ser_cfg: dict):
        self.port = ser_cfg["port"]
        self.baudrate = ser_cfg["baudrate"]
        self.timeout = ser_cfg["timeout"]
        self._ser: Optional[serial.Serial] = None

    def connect(self):
        log.info("Opening serial %s @ %d …", self.port, self.baudrate)
        self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        time.sleep(2)  # let GRBL boot
        self._flush()
        log.info("Serial connection ready.")

    def disconnect(self):
        if self._ser and self._ser.is_open:
            self._ser.close()
            log.info("Serial port closed.")

    def _flush(self):
        if self._ser:
            self._ser.flushInput()

    def send(self, command: str, wait: bool = True, timeout: float = None) -> list[str]:
        """Send a command and collect response lines until 'ok' or timeout."""
        if self._ser is None or not self._ser.is_open:
            raise RuntimeError("Serial port not open.")

        command = command.strip()
        log.info(">> %s", command)
        self._ser.write(f"{command}\n".encode())

        if not wait:
            return []

        effective_timeout = timeout if timeout is not None else self.timeout
        responses: list[str] = []
        deadline = time.time() + effective_timeout
        while time.time() < deadline:
            line = self._ser.readline().decode(errors="replace").strip()
            if not line:
                continue
            log.info("<< %s", line)
            responses.append(line)
            if "ok" in line.lower():
                return responses
            if "error" in line.lower():
                log.error("Controller error: %s", line)
                return responses

        log.warning("Timeout waiting for response after: %s", command)
        return responses

    def home(self, retries: int = 3) -> bool:
        """
        Send $H and verify homing completes successfully.

        Retries up to `retries` times. Returns True if 'ok' was received,
        False if all attempts failed.
        """
        for attempt in range(1, retries + 1):
            log.info("Sending $H homing command (attempt %d/%d) …", attempt, retries)
            self._flush()
            responses = self.send("$H", timeout=120)

            # Check if any response line contains 'ok'
            if any("ok" in line.lower() for line in responses):
                log.info("Homing completed successfully.")
                return True

            # Check for error
            if any("error" in line.lower() for line in responses):
                log.error("Homing returned error: %s", responses)
            else:
                log.warning("Homing timed out or gave no acknowledgement.")

            if attempt < retries:
                log.info("Retrying in 3 seconds …")
                time.sleep(3)

        log.error("Homing failed after %d attempts.", retries)
        return False

    def move_to(self, x: float = None, y: float = None, feed: float = 1000):
        parts = ["G1"]
        if x is not None:
            parts.append(f"X{x:.3f}")
        if y is not None:
            parts.append(f"Y{y:.3f}")
        parts.append(f"F{feed:.0f}")
        self.send(" ".join(parts))
        self.send("G4 P0")  # wait for motion to complete


# ---------------------------------------------------------------------------
# Grid Builder
# ---------------------------------------------------------------------------

def build_grid(grid_cfg: dict) -> list[tuple[float, float]]:
    """Build a serpentine (zig-zag) list of (x, y) positions from config."""
    xs = np.arange(
        grid_cfg["x_start"],
        grid_cfg["x_stop"] + grid_cfg["x_step"] / 2,
        grid_cfg["x_step"],
    )
    ys = np.arange(
        grid_cfg["y_start"],
        grid_cfg["y_stop"] + grid_cfg["y_step"] / 2,
        grid_cfg["y_step"],
    )

    points: list[tuple[float, float]] = []
    for yi, y in enumerate(ys):
        row = xs if yi % 2 == 0 else xs[::-1]
        for x in row:
            points.append((float(x), float(y)))

    return points


# ---------------------------------------------------------------------------
# Main Scan Routine
# ---------------------------------------------------------------------------

def run_scan(cfg: dict):
    """Home → user confirm → grid move + capture → save files."""

    camera = OakDDepthCamera(cfg["camera"])
    gcode = GcodeSender(cfg["serial"])

    base_dir = Path(cfg["output"]["directory"])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = base_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    feedrate = cfg["motion"]["feedrate"]
    settle   = cfg["motion"]["settle_time"]

    save_raw   = cfg["output"]["save_raw_npy"]
    save_depth = cfg["output"]["save_depth_png"]

    try:
        # ── 1. Connect hardware ──
        camera.start()
        gcode.connect()

        # ── 2. Home with $H — retries automatically ──
        if not gcode.home():
            log.error("Homing failed. Aborting scan.")
            return

        # ── 3. Zero machine position and set modes ──
        gcode.send("G92 X0 Y0")  # reset MPos to zero at home
        gcode.send("G90")
        gcode.send("G21")

        # ── 4. Walk the grid ──
        points = build_grid(cfg["grid"])
        total = len(points)
        manifest: list[dict] = []

        log.info("Starting grid scan: %d positions at F%d", total, feedrate)
        print(f"\nScanning {total} grid positions …\n")

        for idx, (x, y) in enumerate(points, start=1):
            log.info("— [%d/%d]  X=%.2f  Y=%.2f", idx, total, x, y)

            gcode.move_to(x=x, y=y, feed=feedrate)
            time.sleep(settle)
            camera.flush_queues()

            # ── Capture ──
            entry: dict = {"index": idx, "x": x, "y": y}

            depth, disparity = camera.capture()
            if depth is None:
                log.warning("No depth frame at position %d — skipping.", idx)
                continue

            tag = f"depth_{idx:04d}_X{x:.1f}_Y{y:.1f}"

            if save_raw:
                raw_path = out_dir / f"{tag}.npy"
                np.save(str(raw_path), depth)
                entry["raw_npy"] = str(raw_path)

            if save_depth:
                cmap = camera.depth_to_colormap(depth)
                png_path = out_dir / f"{tag}.png"
                cv2.imwrite(str(png_path), cmap)
                entry["depth_png"] = str(png_path)

            manifest.append(entry)
            log.info("  Saved: %s", tag)

        # ── 5. Save manifest ──
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        log.info("Manifest saved → %s", manifest_path)

        print(f"\nDone — {len(manifest)} depth maps saved to: {out_dir}/")
        print(f"Manifest: {manifest_path}")

    finally:
        gcode.disconnect()
        camera.stop()


# ---------------------------------------------------------------------------
# Live Preview (no serial needed)
# ---------------------------------------------------------------------------

def run_preview(cfg: dict):
    camera = OakDDepthCamera(cfg["camera"])
    camera.start()
    try:
        log.info("Live preview — press 'q' to quit.")
        while True:
            depth = camera.get_depth_frame()
            if depth is not None:
                cv2.imshow("Depth", camera.depth_to_colormap(depth))
            if cv2.waitKey(1) == ord("q"):
                break
        cv2.destroyAllWindows()
    finally:
        camera.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OAK-D grid depth scanner")
    parser.add_argument(
        "-c", "--config",
        default="scan_config.yaml",
        help="Path to YAML config file (default: scan_config.yaml)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Live depth preview only (no serial / no motion)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.preview:
        run_preview(cfg)
    else:
        run_scan(cfg)


if __name__ == "__main__":
    main()