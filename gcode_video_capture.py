#!/usr/bin/env python3
"""
G-code + OAK-D SR Video Capture (Multi-Pass)
=============================================
1. Homes the machine with $H and waits for acknowledgement.
2. Reads pass geometry and camera settings from vid_config.yaml.
3. For each pass, moves the gantry along a sweep axis at a set speed
   while recording video from the left and right cameras.

Video is MJPEG-encoded on the OAK-D SR's hardware encoder — the host
receives compressed JPEG packets over USB and writes them straight into
an AVI container with zero CPU encoding overhead.

The OAK-D SR has two cameras:
    CAM_B — left
    CAM_C — right

Requirements:
    pip install depthai opencv-python numpy pyserial pyyaml

Usage:
    python gcode_video_capture.py                        # uses vid_config.yaml
    python gcode_video_capture.py -c my_config.yaml      # custom config
    python gcode_video_capture.py --preview               # camera preview only
"""

import argparse
from datetime import datetime
import json
import logging
import queue
import struct
import sys
import threading
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
    """Load and return the YAML config, applying defaults for missing keys."""
    p = Path(path)
    if not p.exists():
        log.error("Config file not found: %s", path)
        sys.exit(1)

    with open(p) as f:
        cfg = yaml.safe_load(f)

    # -- serial --
    cfg.setdefault("serial", {})
    cfg["serial"].setdefault("port", "/dev/ttyUSB0")
    cfg["serial"].setdefault("baudrate", 115200)
    cfg["serial"].setdefault("timeout", 10.0)

    # -- camera --
    cfg.setdefault("camera", {})
    cfg["camera"].setdefault("fps", 30)
    cfg["camera"].setdefault("width", 1280)
    cfg["camera"].setdefault("height", 800)
    cfg["camera"].setdefault("warmup_frames", 30)
    cfg["camera"].setdefault("mjpeg_quality", 95)   # 0–100, hardware encoder

    # -- motion --
    cfg.setdefault("motion", {})
    cfg["motion"].setdefault("feedrate", 1000)
    cfg["motion"].setdefault("return_feedrate", 4000)
    cfg["motion"].setdefault("start_x", 0.0)
    cfg["motion"].setdefault("end_x", 300.0)
    cfg["motion"].setdefault("start_y", 0.0)
    cfg["motion"].setdefault("end_y", 300.0)
    cfg["motion"].setdefault("step_y", 50.0)
    cfg["motion"].setdefault("settle_time", 1.0)

    # -- output --
    cfg.setdefault("output", {})
    cfg["output"].setdefault("directory", "video_output")
    cfg["output"].setdefault("save_left_video", True)
    cfg["output"].setdefault("save_right_video", True)

    return cfg


# ---------------------------------------------------------------------------
# AVI MJPEG Writer — writes raw JPEG bytes into a proper AVI container
# ---------------------------------------------------------------------------

class AviMjpegWriter:
    """Muxes raw JPEG frames into an AVI RIFF container.

    No encoding happens here — each frame is already a complete JPEG image
    produced by the OAK hardware encoder.  The host just wraps them in the
    AVI chunk structure and writes an index at the end.
    """

    def __init__(self, path: str, width: int, height: int, fps: float):
        self._f = open(path, "wb")
        self._w = width
        self._h = height
        self._fps = fps
        self._frame_index: list[tuple[int, int]] = []   # (offset, size)
        self._frame_count = 0

        # Track byte positions that need patching on close
        self._riff_size_pos = 0
        self._movi_size_pos = 0
        self._movi_data_start = 0
        self._avih_frames_pos = 0
        self._strh_length_pos = 0

        self._write_headers()

    def _write_headers(self):
        f = self._f
        pk = struct.pack

        # ── RIFF header ──
        f.write(b"RIFF")
        self._riff_size_pos = f.tell()
        f.write(pk("<I", 0))            # placeholder — patched on close
        f.write(b"AVI ")

        # ── hdrl LIST ──
        f.write(b"LIST")
        hdrl_size_pos = f.tell()
        f.write(pk("<I", 0))            # placeholder
        f.write(b"hdrl")
        hdrl_content_start = f.tell()

        # avih chunk (AVIMAINHEADER — 56 bytes)
        us_per_frame = int(1_000_000 / self._fps)
        f.write(b"avih")
        f.write(pk("<I", 56))
        avih_data_start = f.tell()
        f.write(pk("<I", us_per_frame))  # dwMicroSecPerFrame
        f.write(pk("<I", 0))             # dwMaxBytesPerSec
        f.write(pk("<I", 0))             # dwPaddingGranularity
        f.write(pk("<I", 0x10))          # dwFlags = AVIF_HASINDEX
        self._avih_frames_pos = f.tell()
        f.write(pk("<I", 0))             # dwTotalFrames — patched
        f.write(pk("<I", 0))             # dwInitialFrames
        f.write(pk("<I", 1))             # dwStreams
        f.write(pk("<I", 1_000_000))     # dwSuggestedBufferSize
        f.write(pk("<I", self._w))       # dwWidth
        f.write(pk("<I", self._h))       # dwHeight
        f.write(pk("<4I", 0, 0, 0, 0))  # dwReserved[4]
        assert f.tell() - avih_data_start == 56

        # strl LIST
        f.write(b"LIST")
        strl_size_pos = f.tell()
        f.write(pk("<I", 0))             # placeholder
        f.write(b"strl")
        strl_content_start = f.tell()

        # strh chunk (AVISTREAMHEADER — 56 bytes)
        f.write(b"strh")
        f.write(pk("<I", 56))
        strh_data_start = f.tell()
        f.write(b"vids")                 # fccType
        f.write(b"MJPG")                 # fccHandler
        f.write(pk("<I", 0))             # dwFlags
        f.write(pk("<HH", 0, 0))         # wPriority, wLanguage
        f.write(pk("<I", 0))             # dwInitialFrames
        f.write(pk("<I", 1))             # dwScale
        f.write(pk("<I", int(self._fps)))  # dwRate
        f.write(pk("<I", 0))             # dwStart
        self._strh_length_pos = f.tell()
        f.write(pk("<I", 0))             # dwLength — patched
        f.write(pk("<I", 1_000_000))     # dwSuggestedBufferSize
        f.write(pk("<I", 0xFFFFFFFF))    # dwQuality
        f.write(pk("<I", 0))             # dwSampleSize
        f.write(pk("<hhhh", 0, 0, self._w, self._h))  # rcFrame
        assert f.tell() - strh_data_start == 56

        # strf chunk (BITMAPINFOHEADER — 40 bytes)
        f.write(b"strf")
        f.write(pk("<I", 40))
        f.write(pk("<I", 40))            # biSize
        f.write(pk("<i", self._w))       # biWidth
        f.write(pk("<i", self._h))       # biHeight
        f.write(pk("<HH", 1, 24))        # biPlanes, biBitCount
        f.write(b"MJPG")                 # biCompression
        f.write(pk("<I", self._w * self._h * 3))  # biSizeImage
        f.write(pk("<iiII", 0, 0, 0, 0))  # pels, clr

        # Patch strl LIST size
        strl_end = f.tell()
        f.seek(strl_size_pos)
        f.write(pk("<I", strl_end - strl_content_start + 4))
        f.seek(strl_end)

        # Patch hdrl LIST size
        hdrl_end = f.tell()
        f.seek(hdrl_size_pos)
        f.write(pk("<I", hdrl_end - hdrl_content_start + 4))
        f.seek(hdrl_end)

        # ── movi LIST ──
        f.write(b"LIST")
        self._movi_size_pos = f.tell()
        f.write(pk("<I", 0))             # placeholder
        f.write(b"movi")
        self._movi_data_start = f.tell()

    # -----------------------------------------------------------------

    def write_frame(self, jpeg_data: bytes):
        """Append one JPEG frame to the movi list."""
        f = self._f
        offset = f.tell() - self._movi_data_start
        size = len(jpeg_data)

        f.write(b"00dc")
        f.write(struct.pack("<I", size))
        f.write(jpeg_data)
        if size % 2:                     # AVI chunks must be word-aligned
            f.write(b"\x00")

        self._frame_index.append((offset, size))
        self._frame_count += 1

    def close(self):
        """Write the idx1 index, patch all header sizes, close the file."""
        f = self._f
        pk = struct.pack

        movi_data_end = f.tell()

        # ── idx1 chunk ──
        f.write(b"idx1")
        f.write(pk("<I", len(self._frame_index) * 16))
        for offset, size in self._frame_index:
            f.write(pk("<4sIII", b"00dc", 0x10, offset, size))

        file_end = f.tell()

        # Patch RIFF size
        f.seek(self._riff_size_pos)
        f.write(pk("<I", file_end - 8))

        # Patch movi LIST size
        f.seek(self._movi_size_pos)
        f.write(pk("<I", movi_data_end - self._movi_data_start + 4))

        # Patch avih dwTotalFrames
        f.seek(self._avih_frames_pos)
        f.write(pk("<I", self._frame_count))

        # Patch strh dwLength
        f.seek(self._strh_length_pos)
        f.write(pk("<I", self._frame_count))

        f.close()
        log.info("  AVI closed — %d frames written.", self._frame_count)


# ---------------------------------------------------------------------------
# OAK-D SR Video Camera
# ---------------------------------------------------------------------------

class OakDVideoCamera:
    """Manages the OAK-D SR camera pipeline using the DepthAI v3 API.

    Two modes:
        hardware_encode=True  — MJPEG-encodes on-device, outputs raw JPEG
                                bytes (for recording).
        hardware_encode=False — outputs BGR numpy frames (for preview).

    Available sockets:
        CAM_B — left
        CAM_C — right
    """

    def __init__(self, cam_cfg: dict, hardware_encode: bool = False):
        self.cam_cfg = cam_cfg
        self._hardware_encode = hardware_encode
        self._pipeline: Optional[dai.Pipeline] = None
        self._left_queue = None
        self._right_queue = None

    # ----- pipeline --------------------------------------------------------

    def _build_pipeline(self) -> dai.Pipeline:
        pipeline = dai.Pipeline()
        fps = self.cam_cfg["fps"]
        size = (self.cam_cfg["width"], self.cam_cfg["height"])

        cam_left = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_B,
        )
        cam_right = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_C,
        )

        if self._hardware_encode:
            # Camera → VideoEncoder (on-device MJPEG) → host queue
            quality = self.cam_cfg.get("mjpeg_quality", 95)

            enc_left = pipeline.create(dai.node.VideoEncoder)
            enc_left.setDefaultProfilePreset(
                fps, dai.VideoEncoderProperties.Profile.MJPEG,
            )
            enc_left.setQuality(quality)
            cam_left.requestOutput(
                size, fps=fps, type=dai.ImgFrame.Type.NV12,
            ).link(enc_left.input)
            self._left_queue = enc_left.out.createOutputQueue()

            enc_right = pipeline.create(dai.node.VideoEncoder)
            enc_right.setDefaultProfilePreset(
                fps, dai.VideoEncoderProperties.Profile.MJPEG,
            )
            enc_right.setQuality(quality)
            cam_right.requestOutput(
                size, fps=fps, type=dai.ImgFrame.Type.NV12,
            ).link(enc_right.input)
            self._right_queue = enc_right.out.createOutputQueue()
        else:
            # Camera → BGR frames for cv2.imshow preview
            self._left_queue = cam_left.requestOutput(
                size, fps=fps, type=dai.ImgFrame.Type.BGR888p,
            ).createOutputQueue()
            self._right_queue = cam_right.requestOutput(
                size, fps=fps, type=dai.ImgFrame.Type.BGR888p,
            ).createOutputQueue()

        return pipeline

    # ----- lifecycle -------------------------------------------------------

    def start(self):
        log.info("Starting OAK-D video pipeline …")
        self._pipeline = self._build_pipeline()
        self._pipeline.start()
        log.info("OAK-D pipeline running.")

    def stop(self):
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
            log.info("OAK-D pipeline stopped.")

    # ----- frame accessors (BGR, for preview) ------------------------------

    def get_left_frame(self) -> Optional[np.ndarray]:
        if self._left_queue is None:
            return None
        msg = self._left_queue.get()
        assert isinstance(msg, dai.ImgFrame)
        return msg.getCvFrame()

    def get_right_frame(self) -> Optional[np.ndarray]:
        if self._right_queue is None:
            return None
        msg = self._right_queue.get()
        assert isinstance(msg, dai.ImgFrame)
        return msg.getCvFrame()

    # ----- encoded accessors (JPEG bytes, for recording) -------------------

    def get_left_encoded(self) -> Optional[bytes]:
        if self._left_queue is None:
            return None
        msg = self._left_queue.get()
        return bytes(msg.getData())

    def get_right_encoded(self) -> Optional[bytes]:
        if self._right_queue is None:
            return None
        msg = self._right_queue.get()
        return bytes(msg.getData())

    # ----- utilities -------------------------------------------------------

    def flush_queues(self):
        for q in (self._left_queue, self._right_queue):
            if q is None:
                continue
            while q.has():
                q.get()

    def warmup(self):
        n = self.cam_cfg["warmup_frames"]
        log.info("Warming up camera (%d frames) …", n)
        for _ in range(n):
            if self._left_queue is not None:
                self._left_queue.get()
            if self._right_queue is not None:
                self._right_queue.get()
        log.info("Warm-up complete.")


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
        time.sleep(2)
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
        for attempt in range(1, retries + 1):
            log.info("Sending $H homing command (attempt %d/%d) …", attempt, retries)
            self._flush()
            responses = self.send("$H", timeout=120)
            if any("ok" in line.lower() for line in responses):
                log.info("Homing completed successfully.")
                return True
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
        self.send("G4 P0")


# ---------------------------------------------------------------------------
# Pass recorder  (background-threaded AVI writer)
# ---------------------------------------------------------------------------

class PassRecorder:
    """Receives JPEG byte-pairs from the capture loop via a queue and writes
    them into AVI files on a background thread.  add_frame() is non-blocking."""

    _SENTINEL = object()

    def __init__(
        self,
        out_dir: Path,
        pass_num: int,
        width: int,
        height: int,
        fps: float,
        save_left_video: bool,
        save_right_video: bool,
    ):
        self.pass_num = pass_num
        self._frame_count = 0

        self._left_writer: Optional[AviMjpegWriter] = None
        self._right_writer: Optional[AviMjpegWriter] = None
        self._left_path: Optional[str] = None
        self._right_path: Optional[str] = None

        tag = f"pass_{pass_num:03d}"

        if save_left_video:
            self._left_path = str(out_dir / f"{tag}_left.avi")
            self._left_writer = AviMjpegWriter(self._left_path, width, height, fps)
            log.info("  Left video   → %s", self._left_path)

        if save_right_video:
            self._right_path = str(out_dir / f"{tag}_right.avi")
            self._right_writer = AviMjpegWriter(self._right_path, width, height, fps)
            log.info("  Right video  → %s", self._right_path)

        self._queue: queue.Queue = queue.Queue()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True,
        )
        self._writer_thread.start()

    def _writer_loop(self):
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                return
            left_bytes, right_bytes = item
            if self._left_writer is not None and left_bytes is not None:
                self._left_writer.write_frame(left_bytes)
            if self._right_writer is not None and right_bytes is not None:
                self._right_writer.write_frame(right_bytes)

    def add_frame(
        self,
        left: Optional[bytes],
        right: Optional[bytes],
    ):
        self._queue.put((left, right))
        self._frame_count += 1

    def close(self) -> dict:
        self._queue.put(self._SENTINEL)
        self._writer_thread.join()

        if self._left_writer is not None:
            self._left_writer.close()
        if self._right_writer is not None:
            self._right_writer.close()

        return {
            "pass": self.pass_num,
            "frame_count": self._frame_count,
            "left_video": self._left_path,
            "right_video": self._right_path,
        }


# ---------------------------------------------------------------------------
# Motion-time estimator
# ---------------------------------------------------------------------------

def estimate_pass_time(distance_mm: float, feedrate_mm_min: float) -> float:
    """Return estimated pass duration in seconds."""
    if feedrate_mm_min <= 0:
        return 0.0
    return (distance_mm / feedrate_mm_min) * 60.0


# ---------------------------------------------------------------------------
# Main Video Capture Routine
# ---------------------------------------------------------------------------

def run_video_capture(cfg: dict):
    """Home → repeat (move + record) for N passes → save."""

    # Hardware encoding enabled for recording
    camera = OakDVideoCamera(cfg["camera"], hardware_encode=True)
    gcode = GcodeSender(cfg["serial"])

    motion = cfg["motion"]
    output = cfg["output"]

    start_x = motion["start_x"]
    end_x   = motion["end_x"]

    start_y = motion["start_y"]
    end_y   = motion["end_y"]
    step_y  = motion["step_y"]

    num_passes = int((end_y - start_y) / step_y)
    feedrate   = motion["feedrate"]
    r_feedrate = motion["return_feedrate"]
    settle     = motion["settle_time"]

    sweep_distance = abs(end_x - start_x)
    y_axis_step = np.linspace(
        start_y, end_y, int((end_y - start_y) / step_y) + 1,
    )
    est_time = estimate_pass_time(sweep_distance, feedrate)

    vid_w = cfg["camera"]["width"]
    vid_h = cfg["camera"]["height"]

    # Output directory
    base_dir = Path(output["directory"])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = base_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    try:
        # ── 1. Connect hardware ──
        camera.start()
        camera.warmup()
        gcode.connect()

        # ── 2. Home ──
        if not gcode.home():
            log.error("Homing failed. Aborting.")
            return

        gcode.send("G92 X0 Y0")
        gcode.send("G90")
        gcode.send("G21")

        manifest: list[dict] = []

        print(f"Moving to initial position ({start_x} , {start_y})")
        gcode.move_to(x=start_x, y=start_y, feed=feedrate)
        gcode.send("G4 P0")

        time.sleep(settle)
        camera.flush_queues()
        
        for p in range(1, num_passes + 1):
            current_y = y_axis_step[p - 1]
            print(f"Moving to starting position ({start_x} , {current_y})")
            gcode.move_to(x=start_x, y=current_y, feed=feedrate)
            gcode.send("G4 P0")

            log.info("━━━ Pass %d / %d ━━━", p, num_passes)
            gcode.send("G4 P0")

            time.sleep(settle)
            
            # gcode.send("G4 P0")

            # ── Open recorder ──
            recorder = PassRecorder(
                out_dir=out_dir,
                pass_num=p,
                width=vid_w,
                height=vid_h,
                fps=cfg["camera"]["fps"],
                save_left_video=output["save_left_video"],
                save_right_video=output["save_right_video"],
            )

            # ── Flush stale frames, then start reader threads ──
            camera.flush_queues()
            left_q: queue.Queue = queue.Queue()
            right_q: queue.Queue = queue.Queue()
            stop_readers = threading.Event()

            def _read_left():
                while not stop_readers.is_set():
                    data = camera.get_left_encoded()
                    if data:
                        left_q.put(data)

            def _read_right():
                while not stop_readers.is_set():
                    data = camera.get_right_encoded()
                    if data:
                        right_q.put(data)

            thr_left = threading.Thread(target=_read_left, daemon=True)
            thr_right = threading.Thread(target=_read_right, daemon=True)
            thr_left.start()
            thr_right.start()

            # ── Send move NON-BLOCKING so recording runs during motion ──
            print(f"Video pass to ({end_x} , {current_y})")
            gcode.send(
                f"G1 X{end_x:.3f} Y{current_y:.3f} F{feedrate:.0f}",
                wait=False,
            )

            # ── Record frames while the gantry is moving ──
            frame_count = 0
            t_start = time.time()
            record_duration = est_time + 1.0

            while (time.time() - t_start) < record_duration:
                try:
                    left_data = left_q.get(timeout=0.1)
                    right_data = right_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                recorder.add_frame(left=left_data, right=right_data)
                frame_count += 1

            # ── Stop reader threads ──
            stop_readers.set()
            thr_left.join(timeout=2)
            thr_right.join(timeout=2)

            elapsed = time.time() - t_start
            log.info(
                "  Pass %d complete: %d frames in %.1f s (%.1f fps)",
                p, frame_count, elapsed,
                frame_count / elapsed if elapsed > 0 else 0,
            )

            # Wait for GRBL to confirm the move finished
            gcode.send("G4 P0")

            # Close recorder, collect metadata
            meta = recorder.close()
            meta["elapsed_s"] = round(elapsed, 2)
            meta["feedrate"] = feedrate
            meta["start_x"] = start_x
            meta["start_y"] = current_y
            meta["end_x"] = end_x
            manifest.append(meta)

            print(f"Returning to initial position ({start_x}, {current_y})")
            gcode.move_to(x=start_x, y=current_y, feed=r_feedrate)
            gcode.send("G4 P0")
            
            # time.sleep(settle)
            gcode.send("G4 P0")
            
            
        # ── Save manifest ──
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        log.info("Manifest saved → %s", manifest_path)

        print(f"\nDone — {num_passes} pass(es) recorded to: {out_dir}/")
        print(f"Manifest: {manifest_path}")

    finally:
        gcode.disconnect()
        camera.stop()


# ---------------------------------------------------------------------------
# Live Preview (no serial needed)
# ---------------------------------------------------------------------------

def run_preview(cfg: dict):
    # Preview uses BGR output, no hardware encoding
    camera = OakDVideoCamera(cfg["camera"], hardware_encode=False)
    camera.start()
    try:
        camera.warmup()
        log.info("Live preview — press 'q' to quit.")
        while True:
            left = camera.get_left_frame()
            if left is not None:
                cv2.imshow("Left (CAM_B)", left)

            right = camera.get_right_frame()
            if right is not None:
                cv2.imshow("Right (CAM_C)", right)

            if cv2.waitKey(1) == ord("q"):
                break
        cv2.destroyAllWindows()
    finally:
        camera.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OAK-D SR video capture over gantry passes",
    )
    parser.add_argument(
        "-c", "--config",
        default="vid_config.yaml",
        help="Path to YAML config file (default: vid_config.yaml)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Live camera preview only (no serial / no motion)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.preview:
        run_preview(cfg)
    else:
        run_video_capture(cfg)


if __name__ == "__main__":
    main()