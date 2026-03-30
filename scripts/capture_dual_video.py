#!/usr/bin/env python3
"""
capture_dual_video.py

Record continuously from BOTH cameras until the user stops the program:

- RGB (PiCam Camera Module 3): records an .h264 video via Picamera2
- Thermal (tCam Mini / Lepton): records TRUE radiometric frames (uint16, TLinear ON)
  saved as chunked .npz files containing:
      raw_u16_frames: (N,120,160) uint16
      telemetry_u16_frames: (N,tele_words) uint16
      timestamps: (N,) float64 (epoch seconds)
      meta_json: JSON string w/ settings + per-chunk stats

Stop the program:
- Press Ctrl+C in the terminal (clean shutdown + flush last chunk)

Outputs (created automatically):
  ibis/data/capture_sessions/session_YYYY-MM-DD_HH-MM-SS/
    rgb/rgb_YYYY-MM-DD_HH-MM-SS.h264
    thermal/thermal_chunk_0000.npz
    thermal/thermal_chunk_0001.npz
    ...
    session_meta.txt

Notes:
- RGB records compressed video.
- Thermal saves *radiometric* uint16 frames (Kelvin scaled by res_k), suitable for later conversion to °C.
"""

from __future__ import annotations

import base64
import json
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# --- Picamera2 (RGB) ---
try:
    from picamera2 import Picamera2
    from picamera2.encoders import H264Encoder
    from picamera2.outputs import FileOutput
    HAS_PICAMERA2 = True
except Exception:
    Picamera2 = None
    H264Encoder = None
    FileOutput = None
    HAS_PICAMERA2 = False

# --- Your tCam path helper / SDK ---
from ibis.tCamSource_path import add_tcam_to_path
add_tcam_to_path()
from tcam import TCam  # noqa: E402


# ----------------------------
# Config
# ----------------------------
H, W = 120, 160

RAD_BASE = 0x4E00
CMD_TLINEAR_ENABLE_SET  = RAD_BASE | 0xC1
CMD_TLINEAR_RES_SET     = RAD_BASE | 0xC5
CMD_TLINEAR_AUTORES_SET = RAD_BASE | 0xC9


@dataclass
class Config:
    # Data root inside your repo
    data_root_rel: Path = Path("ibis") / "data"
    session_prefix: str = "session"

    # RGB
    rgb_fps: int = 30
    rgb_bitrate: int = 12_000_000
    rgb_video_size: tuple[int, int] = (1920, 1080)  # sane default for sustained video
    rgb_prefix: str = "rgb"

    # Thermal
    thermal_target_fps: float = 8.0
    thermal_chunk_frames: int = 300   # ~37s @ 8 fps
    thermal_prefix: str = "thermal_chunk"

    # Thermal camera settings (match your single-frame script defaults)
    emissivity_percent: int = 95
    gain_mode: int = 0
    run_ffc_on_start: bool = True
    res_k: float = 0.01  # 0.01K resolution


# ----------------------------
# Small helpers (from your script)
# ----------------------------
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _b64_to_u16(b64_str: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64_str), dtype="<u2")


def _parse_telemetry(tele_u16: np.ndarray) -> dict:
    if tele_u16.size < 240:
        return {"tlinear_enabled": -1, "tlinear_res_k": None}

    tlinear_enabled = int(tele_u16[208])
    res_flag = int(tele_u16[209])  # 0=>0.1K, 1=>0.01K
    res_k = 0.01 if res_flag == 1 else 0.1

    spot_k = tele_u16[210] * res_k
    spot_c = spot_k - 273.15

    return {
        "tlinear_enabled": tlinear_enabled,
        "tlinear_res_k": float(res_k),
        "spot_c": float(spot_c),
        "fpa_c": float(tele_u16[24] / 100.0 - 273.15),
        "housing_c": float(tele_u16[26] / 100.0 - 273.15),
    }


# ----------------------------
# Repo root inference
# ----------------------------
def repo_root_from_this_file() -> Path:
    # assumes this file is in ~/projects/IBIS/scripts/
    return Path(__file__).resolve().parents[1]


# ----------------------------
# Stop flag
# ----------------------------
class Stopper:
    def __init__(self) -> None:
        self._stop = False

    def request(self) -> None:
        self._stop = True

    def should_stop(self) -> bool:
        return self._stop


# ----------------------------
# RGB recorder
# ----------------------------
class RGBRecorder:
    def __init__(self, out_path: Path, cfg: Config) -> None:
        if not HAS_PICAMERA2:
            raise RuntimeError(
                "picamera2 is not available. Install via apt:\n"
                "  sudo apt update\n"
                "  sudo apt install -y python3-picamera2\n"
                "If you're in a venv, recreate with --system-site-packages."
            )
        self.out_path = out_path
        self.cfg = cfg
        self.picam2 = Picamera2()
        self.encoder: Optional[H264Encoder] = None

    def start(self) -> None:
        # Use a video configuration (stable). Full sensor res video is heavy; start with 1080p.
        video_cfg = self.picam2.create_video_configuration(
            main={"size": self.cfg.rgb_video_size, "format": "RGB888"},
            controls={"FrameRate": self.cfg.rgb_fps},
        )
        self.picam2.configure(video_cfg)

        self.encoder = H264Encoder(bitrate=self.cfg.rgb_bitrate)
        self.picam2.start_recording(self.encoder, FileOutput(str(self.out_path)))

    def stop(self) -> None:
        try:
            self.picam2.stop_recording()
        except Exception:
            pass
        try:
            self.picam2.close()
        except Exception:
            pass


# ----------------------------
# Thermal streaming capture (based on your current tCam code)
# ----------------------------
class ThermalStreamer:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.cam = TCam(is_hw=True)

        stat = self.cam.connect()
        if stat.get("status") != "connected":
            raise RuntimeError(f"Could not connect tCam: {stat}")

        # Normal operating settings
        self.cam.set_config(
            emissivity=int(cfg.emissivity_percent),
            gain_mode=int(cfg.gain_mode),
            agc_enabled=0,
        )

        # TLinear ON, autores OFF, resolution set
        self.cam.set_lep_cci(CMD_TLINEAR_AUTORES_SET, [0])
        self.cam.set_lep_cci(CMD_TLINEAR_RES_SET, [1 if abs(cfg.res_k - 0.01) < 1e-6 else 0])
        self.cam.set_lep_cci(CMD_TLINEAR_ENABLE_SET, [1])

        if cfg.run_ffc_on_start:
            self.cam.run_ffc()

    def get_frame(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Returns:
          raw_u16 (120x160) uint16
          tele_u16 (Nwords,) uint16
          tele_summary (dict)
        """
        frame = self.cam.get_image()

        raw_words = _b64_to_u16(frame["radiometric"])
        if raw_words.size != H * W:
            raise RuntimeError(f"Unexpected radiometric size: {raw_words.size} words (expected {H*W})")
        raw_u16 = raw_words.reshape((H, W))

        tele_u16 = _b64_to_u16(frame["telemetry"])
        tele = _parse_telemetry(tele_u16)

        # Hard checks (but keep it robust for streaming)
        if tele.get("tlinear_enabled") != 1:
            raise RuntimeError(f"TLinear not enabled (telemetry says {tele.get('tlinear_enabled')}).")

        # Median sanity check (your original threshold)
        med = float(np.median(raw_u16))
        if not (20000 <= med <= 40000):
            raise RuntimeError(f"Frame does not look Kelvin-scaled (median={med:.1f}).")

        return raw_u16, tele_u16, tele

    def shutdown(self) -> None:
        try:
            self.cam.shutdown()
        except Exception:
            pass


class ThermalChunkWriter:
    def __init__(self, out_dir: Path, cfg: Config) -> None:
        self.out_dir = out_dir
        self.cfg = cfg
        self.chunk_idx = 0

        self.raw_frames: list[np.ndarray] = []
        self.tele_frames: list[np.ndarray] = []
        self.times: list[float] = []

        self.last_tele_summary: Optional[dict] = None

    def add(self, raw_u16: np.ndarray, tele_u16: np.ndarray, tele_summary: dict, t: float) -> None:
        self.raw_frames.append(raw_u16)
        self.tele_frames.append(tele_u16)
        self.times.append(t)
        self.last_tele_summary = tele_summary

    def flush_if_needed(self) -> Optional[Path]:
        if len(self.raw_frames) < self.cfg.thermal_chunk_frames:
            return None
        return self._write_chunk()

    def close(self) -> Optional[Path]:
        if not self.raw_frames:
            return None
        return self._write_chunk()

    def _write_chunk(self) -> Path:
        out_path = self.out_dir / f"{self.cfg.thermal_prefix}_{self.chunk_idx:04d}.npz"

        raw_stack = np.stack(self.raw_frames, axis=0).astype(np.uint16)

        # Telemetry words length can vary; stack by padding to max length in chunk
        max_len = max(t.size for t in self.tele_frames)
        tele_stack = np.zeros((len(self.tele_frames), max_len), dtype=np.uint16)
        for i, t in enumerate(self.tele_frames):
            tele_stack[i, : t.size] = t.astype(np.uint16)

        # Basic per-chunk stats
        med = float(np.median(raw_stack))
        meta = {
            "chunk_index": int(self.chunk_idx),
            "captured_frames": int(raw_stack.shape[0]),
            "shape_hw": [H, W],
            "emissivity_percent": int(self.cfg.emissivity_percent),
            "gain_mode": int(self.cfg.gain_mode),
            "agc_enabled": 0,
            "tlinear_res_k_requested": float(self.cfg.res_k),
            "telemetry_last": self.last_tele_summary or {},
            "raw_min": int(raw_stack.min()),
            "raw_max": int(raw_stack.max()),
            "raw_median": med,
        }

        np.savez_compressed(
            out_path,
            raw_u16_frames=raw_stack,
            telemetry_u16_frames=tele_stack,
            timestamps=np.array(self.times, dtype=np.float64),
            meta_json=json.dumps(meta),
        )

        # Reset buffers
        self.raw_frames.clear()
        self.tele_frames.clear()
        self.times.clear()
        self.chunk_idx += 1

        return out_path


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    cfg = Config()
    repo_root = repo_root_from_this_file()
    data_root = repo_root / cfg.data_root_rel

    session_dir = data_root / "capture_sessions" / f"{cfg.session_prefix}_{_ts()}"
    rgb_dir = session_dir / "rgb"
    thermal_dir = session_dir / "thermal"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    thermal_dir.mkdir(parents=True, exist_ok=True)

    # Metadata
    (session_dir / "session_meta.txt").write_text(
        "\n".join(
            [
                f"started: {datetime.now().isoformat()}",
                f"session_dir: {session_dir}",
                f"rgb_fps: {cfg.rgb_fps}",
                f"rgb_bitrate: {cfg.rgb_bitrate}",
                f"rgb_video_size: {cfg.rgb_video_size}",
                f"thermal_target_fps: {cfg.thermal_target_fps}",
                f"thermal_chunk_frames: {cfg.thermal_chunk_frames}",
                f"emissivity_percent: {cfg.emissivity_percent}",
                f"gain_mode: {cfg.gain_mode}",
                f"res_k: {cfg.res_k}",
                f"run_ffc_on_start: {cfg.run_ffc_on_start}",
            ]
        )
        + "\n"
    )

    stop = Stopper()

    def _sig_handler(sig, frame) -> None:
        stop.request()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # Start RGB recording
    rgb_path = rgb_dir / f"{cfg.rgb_prefix}_{_ts()}.h264"
    rgb = None
    try:
        rgb = RGBRecorder(rgb_path, cfg)
        rgb.start()
    except Exception as e:
        print(f"[ERROR] RGB start failed: {e}", file=sys.stderr)
        return 1

    # Start thermal streaming
    try:
        tcam = ThermalStreamer(cfg)
    except Exception as e:
        print(f"[ERROR] Thermal init failed: {e}", file=sys.stderr)
        if rgb:
            rgb.stop()
        return 1

    writer = ThermalChunkWriter(thermal_dir, cfg)

    print("\n=== Dual capture started ===")
    print(f"Session folder: {session_dir}")
    print(f"RGB video:      {rgb_path}")
    print(f"Thermal chunks: {thermal_dir} / {cfg.thermal_prefix}_####.npz")
    print("\nStop anytime with Ctrl+C.\n")

    # Timing loop (thermal)
    interval = 1.0 / max(cfg.thermal_target_fps, 0.1)
    next_deadline = time.time()

    frames_ok = 0
    frames_fail = 0
    last_print = time.time()

    try:
        while not stop.should_stop():
            now = time.time()

            # Capture one thermal frame
            try:
                raw_u16, tele_u16, tele = tcam.get_frame()
                writer.add(raw_u16, tele_u16, tele, now)
                frames_ok += 1

                written = writer.flush_if_needed()
                if written is not None:
                    print(f"[THERM] wrote {written.name}")

            except Exception as e:
                frames_fail += 1
                # Keep going unless it's a persistent failure; brief backoff
                time.sleep(0.05)

            # Status line every ~5 seconds
            if time.time() - last_print > 5.0:
                print(f"[STAT] thermal ok={frames_ok} fail={frames_fail} | ctrl+c to stop")
                last_print = time.time()

            # Maintain target thermal rate (without busy-wait)
            next_deadline += interval
            sleep_time = next_deadline - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_deadline = time.time()

    finally:
        print("\nStopping capture...")

        # Flush remaining thermal frames
        try:
            last_chunk = writer.close()
            if last_chunk:
                print(f"[THERM] wrote {last_chunk.name}")
        except Exception:
            pass

        # Shutdown thermal camera
        try:
            tcam.shutdown()
        except Exception:
            pass

        # Stop RGB
        try:
            if rgb:
                rgb.stop()
        except Exception:
            pass

        print("Capture stopped.")
        print(f"Session saved at: {session_dir}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
