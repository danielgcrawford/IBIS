#!/usr/bin/env python3
"""
capture_picam3_save.py

Minimal PiCam Camera Module 3 capture script (Picamera2).
- Captures a *full-resolution* still image (max sensor resolution)
- Saves it into a local "raw" folder (created if missing)
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

from picamera2 import Picamera2


def capture_picam3_fullres_to_raw(
    raw_dir: str | Path = "data/RGB_raw",
    prefix: str = "picam3",
    file_ext: str = "jpg",
) -> Path:
    """
    Capture the highest-resolution still image available from the PiCam 3
    and save it into `raw_dir`.

    Returns:
        Path to the saved image file.

    Notes:
      - Uses Picamera2 still configuration at the sensor's maximum resolution.
      - Saves as PNG by default. (JPG also works: file_ext="jpg")
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Timestamped filename like: picam3_2026-02-06_13-05-22.jpg
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = raw_dir / f"{prefix}_{ts}.{file_ext.lstrip('.')}"

    picam2 = Picamera2()
    try:
        # Determine the maximum sensor resolution and configure for a still capture.
        sensor_res = picam2.sensor_resolution  # (width, height)
        still_config = picam2.create_still_configuration(
            main={"size": sensor_res},  # full resolution
            buffer_count=2,
        )
        picam2.configure(still_config)

        # Start the camera, give it a moment to settle, then capture directly to file.
        picam2.start()
        picam2.capture_file(str(out_path))
    finally:
        # Always release the camera cleanly.
        try:
            picam2.stop()
        except Exception:
            pass
        picam2.close()

    return out_path


if __name__ == "__main__":
    saved_path = capture_picam3_fullres_to_raw(raw_dir="data/RGB_raw", prefix="picam3", file_ext="png")
    print(f"Saved: {saved_path}")
