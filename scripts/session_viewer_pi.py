#!/usr/bin/env python3
"""
session_viewer_pi.py (Streamlit) — run on the Raspberry Pi

Loads capture sessions saved by capture_dual_video.py under:
  ibis/data/capture_sessions/session_YYYY-MM-DD_HH-MM-SS/

Session structure expected:
  session_xxx/
    rgb/*.h264 (optional) and/or rgb/*.mp4
    thermal/thermal_chunk_####.npz

Features (best preliminary pre-orthomosaic workflow):
- Thermal QA summary (duration, fps estimate, percentiles)
- Thermal frame browser (radiometric -> Celsius)
- ROI temperature trend vs time (mean/min/max)
- RGB handling:
    - If .mp4 exists: play in Streamlit
    - If only .h264 exists: provide a button to convert to rgb.mp4 via ffmpeg

Run (on Pi):
  source .venv/bin/activate
  streamlit run scripts/session_viewer_pi.py --server.address 0.0.0.0 --server.port 8501

Then open from your laptop browser:
  http://<PI_IP_ADDRESS>:8501
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


H, W = 120, 160


# ----------------------------
# Paths
# ----------------------------
def repo_root_from_this_file() -> Path:
    # assumes scripts/ is directly under repo root
    return Path(__file__).resolve().parents[1]


def capture_sessions_root() -> Path:
    return repo_root_from_this_file() / "ibis" / "data" / "capture_sessions"


def list_sessions(root: Path) -> List[Path]:
    if not root.exists():
        return []
    # session_YYYY... folders
    sessions = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("session_")], reverse=True)
    return sessions


def find_rgb_files(rgb_dir: Path) -> Tuple[List[Path], List[Path]]:
    h264 = sorted(rgb_dir.glob("*.h264"))
    mp4 = sorted(rgb_dir.glob("*.mp4"))
    return h264, mp4


def find_thermal_chunks(thermal_dir: Path) -> List[Path]:
    return sorted(thermal_dir.glob("thermal_chunk_*.npz"))


# ----------------------------
# Thermal loading / conversion
# ----------------------------
@st.cache_data(show_spinner=False)
def load_thermal_chunks(npz_paths: List[str]) -> Tuple[np.ndarray, np.ndarray, dict]:
    frames_list = []
    times_list = []
    meta_last = {}

    for p in npz_paths:
        z = np.load(p, allow_pickle=False)
        frames_list.append(z["raw_u16_frames"].astype(np.uint16))
        times_list.append(z["timestamps"].astype(np.float64))
        meta_last = json.loads(str(z["meta_json"]))

    frames_u16 = np.concatenate(frames_list, axis=0) if frames_list else np.zeros((0, H, W), dtype=np.uint16)
    times = np.concatenate(times_list, axis=0) if times_list else np.zeros((0,), dtype=np.float64)
    return frames_u16, times, meta_last


def raw_to_celsius(frames_u16: np.ndarray, res_k: float) -> np.ndarray:
    """TLinear Kelvin = raw * res_k; Celsius = Kelvin - 273.15"""
    return frames_u16.astype(np.float32) * float(res_k) - 273.15


def to_8bit_display(frame_c: np.ndarray) -> np.ndarray:
    """Percentile stretch to 8-bit for easy viewing."""
    lo, hi = np.percentile(frame_c, [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    x = np.clip((frame_c - lo) / (hi - lo), 0, 1)
    return (x * 255).astype(np.uint8)


def compute_fps(times: np.ndarray) -> Tuple[float, np.ndarray]:
    if len(times) < 2:
        return 0.0, np.array([])
    dt = np.diff(times)
    fps_inst = np.where(dt > 0, 1.0 / dt, 0.0)
    fps_mean = float(np.mean(fps_inst))
    return fps_mean, fps_inst


def roi_stats(frames_c: np.ndarray, x0: int, x1: int, y0: int, y1: int):
    roi = frames_c[:, y0:y1, x0:x1]
    mean = roi.mean(axis=(1, 2))
    mn = roi.min(axis=(1, 2))
    mx = roi.max(axis=(1, 2))
    return mean, mn, mx


# ----------------------------
# RGB conversion (ffmpeg)
# ----------------------------
def ffmpeg_exists() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=False)
        return True
    except Exception:
        return False


def convert_h264_to_mp4(h264_path: Path, fps: int, out_mp4: Optional[Path] = None) -> Path:
    """
    Convert raw .h264 bitstream to a browser-friendly MP4.

    Note: We encode (libx264) for compatibility.
    """
    if out_mp4 is None:
        out_mp4 = h264_path.with_suffix(".mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(int(fps)),
        "-i", str(h264_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr}")
    return out_mp4


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="IBIS Session Viewer (Pi)", layout="wide")
st.title("IBIS Session Viewer (on Raspberry Pi)")

st.markdown(
    """
This viewer runs on the Pi and reads session folders directly from:
`ibis/data/capture_sessions/`

**Best early workflow before orthomosaics:**
- Confirm thermal is valid (radiometric → °C)
- Check timing (fps stability, gaps)
- Extract ROI temperature trends
- Convert RGB `.h264` → `.mp4` so it plays in the browser
"""
)

root = capture_sessions_root()
sessions = list_sessions(root)

if not sessions:
    st.error(f"No sessions found in: {root}")
    st.stop()

session_names = [s.name for s in sessions]

st.subheader("1) Choose a session")
chosen = st.selectbox("Session folder", session_names, index=0)
session_dir = root / chosen

rgb_dir = session_dir / "rgb"
thermal_dir = session_dir / "thermal"

if not rgb_dir.exists() or not thermal_dir.exists():
    st.error("Chosen folder does not contain required subfolders: rgb/ and thermal/")
    st.stop()

h264_files, mp4_files = find_rgb_files(rgb_dir)
chunks = find_thermal_chunks(thermal_dir)

st.write(f"RGB: {len(h264_files)} .h264 | {len(mp4_files)} .mp4")
st.write(f"Thermal: {len(chunks)} chunk files")

if not chunks:
    st.error("No thermal_chunk_####.npz files found in thermal/.")
    st.stop()

st.subheader("2) Thermal load settings")
colA, colB, colC = st.columns([1, 1, 1])

with colA:
    max_frames_load = st.number_input("Max thermal frames to load (0 = all)", min_value=0, value=0, step=1000)

with colB:
    res_k_override = st.selectbox("TLinear resolution", ["Use telemetry", "0.01K", "0.1K"], index=0)

with colC:
    rgb_fps_for_convert = st.number_input("RGB FPS (for conversion)", min_value=1, value=30, step=1)

with st.spinner("Loading thermal chunks..."):
    frames_u16, times, meta_last = load_thermal_chunks([str(p) for p in chunks])

if max_frames_load and frames_u16.shape[0] > max_frames_load:
    frames_u16 = frames_u16[:max_frames_load]
    times = times[:max_frames_load]

# Determine res_k
if res_k_override == "Use telemetry":
    res_k = float(meta_last.get("telemetry_last", {}).get("tlinear_res_k", meta_last.get("tlinear_res_k_requested", 0.01)))
else:
    res_k = 0.01 if res_k_override == "0.01K" else 0.1

frames_c = raw_to_celsius(frames_u16, res_k)
fps_mean, fps_inst = compute_fps(times)

st.success(f"Loaded thermal frames: {frames_c.shape[0]} | res_k={res_k} | mean fps≈{fps_mean:.2f}")

st.subheader("3) Session summary")
duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
p1, p50, p99 = np.percentile(frames_c, [1, 50, 99])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Thermal frames", f"{frames_c.shape[0]}")
c2.metric("Duration (s)", f"{duration:.1f}")
c3.metric("Mean thermal fps", f"{fps_mean:.2f}")
c4.metric("Thermal chunks", f"{len(chunks)}")
st.write(f"Thermal °C percentiles (1/50/99): {p1:.2f}, {p50:.2f}, {p99:.2f}")

if fps_inst.size:
    fig = plt.figure()
    plt.plot(fps_inst)
    plt.title("Instantaneous Thermal FPS (1/dt)")
    plt.xlabel("Frame index")
    plt.ylabel("FPS")
    st.pyplot(fig)

st.subheader("4) RGB video")
if mp4_files:
    st.write("MP4 found:", mp4_files[0].name)
    st.video(str(mp4_files[0]))
else:
    if not h264_files:
        st.warning("No RGB video found (.h264 or .mp4).")
    else:
        st.write("H264 found:", h264_files[0].name)
        if not ffmpeg_exists():
            st.error("ffmpeg is not installed. Run: sudo apt install -y ffmpeg")
        else:
            if st.button("Convert .h264 → rgb.mp4"):
                out_mp4 = rgb_dir / "rgb.mp4"
                try:
                    mp4_path = convert_h264_to_mp4(h264_files[0], fps=int(rgb_fps_for_convert), out_mp4=out_mp4)
                    st.success(f"Saved: {mp4_path}")
                    st.video(str(mp4_path))
                except Exception as e:
                    st.error(str(e))

st.subheader("5) Thermal frame browser")
idx = st.slider("Thermal frame index", min_value=0, max_value=max(frames_c.shape[0] - 1, 0), value=0)
frame = frames_c[idx]
disp_u8 = to_8bit_display(frame)

col1, col2 = st.columns([1, 1])
with col1:
    st.image(disp_u8, caption=f"Thermal preview (8-bit stretched) | frame {idx}", clamp=True)
with col2:
    fig2 = plt.figure()
    plt.imshow(frame, interpolation="nearest")
    plt.colorbar(label="°C")
    plt.title("Thermal (°C)")
    st.pyplot(fig2)

st.subheader("6) ROI temperature trend (most useful early analysis)")
st.caption("Pick a rectangle ROI in thermal pixel coordinates (160x120).")

r1, r2, r3, r4 = st.columns(4)
x0 = int(r1.number_input("x0", min_value=0, max_value=W - 1, value=0))
x1 = int(r2.number_input("x1", min_value=1, max_value=W, value=W))
y0 = int(r3.number_input("y0", min_value=0, max_value=H - 1, value=0))
y1 = int(r4.number_input("y1", min_value=1, max_value=H, value=H))

x0, x1 = min(x0, x1 - 1), max(x0 + 1, x1)
y0, y1 = min(y0, y1 - 1), max(y0 + 1, y1)

mean, mn, mx = roi_stats(frames_c, x0, x1, y0, y1)

fig3 = plt.figure()
plt.plot(mean, label="mean")
plt.plot(mn, label="min", alpha=0.7)
plt.plot(mx, label="max", alpha=0.7)
plt.title(f"ROI temp trend | x[{x0}:{x1}] y[{y0}:{y1}]")
plt.xlabel("Frame index")
plt.ylabel("°C")
plt.legend()
st.pyplot(fig3)

st.caption(f"Session folder on Pi: {session_dir}")
