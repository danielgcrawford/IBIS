# app/pages/1_Session_Viewer.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

H, W = 120, 160


# ---------- Path helpers ----------
def repo_root() -> Path:
    # pages/ -> app/ -> repo root
    return Path(__file__).resolve().parents[2]


def sessions_root() -> Path:
    # hosted demo data lives here
    return repo_root() / "sample_data" / "capture_sessions"


def list_sessions(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("session_")], reverse=True)


def find_rgb_mp4(rgb_dir: Path) -> Path | None:
    mp4s = sorted(rgb_dir.glob("*.mp4"))
    return mp4s[0] if mp4s else None


def find_thermal_chunks(thermal_dir: Path) -> List[Path]:
    return sorted(thermal_dir.glob("thermal_chunk_*.npz"))


# ---------- Thermal loading ----------
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
    return frames_u16.astype(np.float32) * float(res_k) - 273.15


def to_8bit_display(frame_c: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(frame_c, [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    x = np.clip((frame_c - lo) / (hi - lo), 0, 1)
    return (x * 255).astype(np.uint8)


def compute_fps(times: np.ndarray):
    if len(times) < 2:
        return 0.0, np.array([])
    dt = np.diff(times)
    fps_inst = np.where(dt > 0, 1.0 / dt, 0.0)
    return float(np.mean(fps_inst)), fps_inst


def roi_stats(frames_c: np.ndarray, x0: int, x1: int, y0: int, y1: int):
    roi = frames_c[:, y0:y1, x0:x1]
    return roi.mean(axis=(1, 2)), roi.min(axis=(1, 2)), roi.max(axis=(1, 2))


# ---------- UI ----------
st.set_page_config(page_title="IBIS — Session Viewer", layout="wide")
st.title("Session Viewer")

root = sessions_root()
sessions = list_sessions(root)

if not sessions:
    st.error(f"No sessions found in: {root}")
    st.stop()

session_names = [s.name for s in sessions]
chosen = st.selectbox("Choose a session", session_names, index=0)
session_dir = root / chosen

rgb_dir = session_dir / "rgb"
thermal_dir = session_dir / "thermal"

rgb_mp4 = find_rgb_mp4(rgb_dir)
chunks = find_thermal_chunks(thermal_dir)

st.caption(f"Session path: `{session_dir}`")
st.write(f"RGB mp4: {'found' if rgb_mp4 else 'not found'}")
st.write(f"Thermal chunks: {len(chunks)}")

if not chunks:
    st.error("No thermal_chunk_####.npz files found.")
    st.stop()

colA, colB = st.columns(2)
with colA:
    max_frames = st.number_input("Max thermal frames to load (0=all)", min_value=0, value=0, step=1000)
with colB:
    res_opt = st.selectbox("TLinear resolution", ["Use telemetry", "0.01K", "0.1K"], index=0)

with st.spinner("Loading thermal chunks..."):
    frames_u16, times, meta_last = load_thermal_chunks([str(p) for p in chunks])

if max_frames and frames_u16.shape[0] > max_frames:
    frames_u16 = frames_u16[:max_frames]
    times = times[:max_frames]

if res_opt == "Use telemetry":
    res_k = float(meta_last.get("telemetry_last", {}).get("tlinear_res_k", meta_last.get("tlinear_res_k_requested", 0.01)))
else:
    res_k = 0.01 if res_opt == "0.01K" else 0.1

frames_c = raw_to_celsius(frames_u16, res_k)
fps_mean, fps_inst = compute_fps(times)

# Summary
st.subheader("Summary")
duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
p1, p50, p99 = np.percentile(frames_c, [1, 50, 99])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Frames", f"{frames_c.shape[0]}")
c2.metric("Duration (s)", f"{duration:.1f}")
c3.metric("Mean FPS", f"{fps_mean:.2f}")
c4.metric("Chunks", f"{len(chunks)}")

st.write(f"Thermal °C percentiles (1/50/99): {p1:.2f}, {p50:.2f}, {p99:.2f}")

if fps_inst.size:
    fig = plt.figure()
    plt.plot(fps_inst)
    plt.title("Instantaneous Thermal FPS (1/dt)")
    plt.xlabel("Frame index")
    plt.ylabel("FPS")
    st.pyplot(fig)

# RGB playback
st.subheader("RGB")
if rgb_mp4:
    st.video(str(rgb_mp4))
else:
    st.warning("No MP4 found in this session (streamlit plays mp4 best).")

# Thermal frame browser
st.subheader("Thermal Frame Browser")
idx = st.slider("Frame index", 0, max(frames_c.shape[0] - 1, 0), 0)
frame = frames_c[idx]

col1, col2 = st.columns([1, 1])
with col1:
    st.image(to_8bit_display(frame), caption=f"Thermal preview (8-bit) | frame {idx}", clamp=True)
with col2:
    fig2 = plt.figure()
    plt.imshow(frame, interpolation="nearest")
    plt.colorbar(label="°C")
    plt.title("Thermal (°C)")
    st.pyplot(fig2)

# ROI trend
st.subheader("ROI Temperature Trend")
r1, r2, r3, r4 = st.columns(4)
x0 = int(r1.number_input("x0", 0, W - 1, 0))
x1 = int(r2.number_input("x1", 1, W, W))
y0 = int(r3.number_input("y0", 0, H - 1, 0))
y1 = int(r4.number_input("y1", 1, H, H))

x0, x1 = min(x0, x1 - 1), max(x0 + 1, x1)
y0, y1 = min(y0, y1 - 1), max(y0 + 1, y1)

mean, mn, mx = roi_stats(frames_c, x0, x1, y0, y1)

fig3 = plt.figure()
plt.plot(mean, label="mean")
plt.plot(mn, label="min", alpha=0.7)
plt.plot(mx, label="max", alpha=0.7)
plt.title(f"ROI x[{x0}:{x1}] y[{y0}:{y1}]")
plt.xlabel("Frame index")
plt.ylabel("°C")
plt.legend()
st.pyplot(fig3)