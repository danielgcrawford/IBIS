
#!/usr/bin/env python3
"""
Minimal tCam capture tool (TLinear ON, true per-pixel temps).

What you get:
1) capture_and_save_raw() -> saves a single .npz bundle (raw_u16 + telemetry_u16 + meta)
2) save_celsius_csv()     -> converts bundle -> full-res 120x160 CSV (°C)
3) save_celsius_png()     -> converts bundle -> nice PNG w/ °C colorbar + min/max

When done testing and want max storage efficiency:
- Comment out ONE LINE each in __main__ for CSV and PNG.
"""

import base64, json, os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from ibis.tCamSource_path import add_tcam_to_path
add_tcam_to_path()

from tcam import TCam

H, W = 120, 160

# Lepton RAD CCI commands (wrapper uses 0x4E00 | cmd_id pattern)
RAD_BASE = 0x4E00
CMD_TLINEAR_ENABLE_SET  = RAD_BASE | 0xC1
CMD_TLINEAR_RES_SET     = RAD_BASE | 0xC5
CMD_TLINEAR_AUTORES_SET = RAD_BASE | 0xC9


def _b64_to_u16(b64_str: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64_str), dtype="<u2")


def _parse_telemetry(tele_u16: np.ndarray) -> dict:
    """Parse the few telemetry fields you care about (matches developer example indices)."""
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


def _raw_to_celsius(raw_u16_img: np.ndarray, res_k: float) -> np.ndarray:
    """TLinear per-pixel: Kelvin = raw * res_k, Celsius = Kelvin - 273.15."""
    return raw_u16_img.astype(np.float32) * float(res_k) - 273.15


def capture_and_save_raw(outdir=".", prefix="tcam", emissivity_percent=95, gain_mode=0,
                         run_ffc=True, res_k=0.01, save_json=False) -> str:
    """
    Capture ONE radiometric frame (TLinear ON) and save:
      <prefix>_<timestamp>_bundle.npz

    Bundle contents:
      raw_u16: (120,160) uint16  (Kelvin scaled by res_k)
      telemetry_u16: uint16 array (usually 240 words)
      meta_json: JSON string with small metadata + telemetry summary
    """
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(outdir, f"{prefix}_{ts}")

    cam = TCam(is_hw=True)
    stat = cam.connect()
    if stat.get("status") != "connected":
        raise RuntimeError(f"Could not connect: {stat}")

    # Normal operating settings (leaves ~0.95 emissivity)
    cam.set_config(emissivity=int(emissivity_percent), gain_mode=int(gain_mode), agc_enabled=0)

    # TLinear ON, autores OFF, resolution set
    cam.set_lep_cci(CMD_TLINEAR_AUTORES_SET, [0])
    cam.set_lep_cci(CMD_TLINEAR_RES_SET, [1 if abs(res_k - 0.01) < 1e-6 else 0])
    cam.set_lep_cci(CMD_TLINEAR_ENABLE_SET, [1])

    if run_ffc:
        cam.run_ffc()

    frame = cam.get_image()
    cam.shutdown()

    if save_json:
        with open(base + "_frame.json", "w") as f:
            json.dump(frame, f, indent=2)

    raw_words = _b64_to_u16(frame["radiometric"])
    if raw_words.size != H * W:
        raise RuntimeError(f"Unexpected radiometric size: {raw_words.size} words (expected {H*W})")
    raw_u16 = raw_words.reshape((H, W))

    tele_u16 = _b64_to_u16(frame["telemetry"])
    tele = _parse_telemetry(tele_u16)

    # Hard safety checks: refuse to proceed if not truly radiometric
    if tele.get("tlinear_enabled") != 1:
        raise RuntimeError(f"TLinear not enabled (telemetry says {tele.get('tlinear_enabled')}).")
    med = float(np.median(raw_u16))
    if not (20000 <= med <= 40000):
        raise RuntimeError(f"Frame does not look like Kelvin-scaled data (median={med:.1f}).")

    meta = {
        "timestamp": ts,
        "emissivity_percent": int(emissivity_percent),
        "gain_mode": int(gain_mode),
        "agc_enabled": 0,
        "telemetry": tele,
        "raw_min": int(raw_u16.min()),
        "raw_max": int(raw_u16.max()),
        "raw_median": float(med),
    }

    bundle_path = base + "_bundle.npz"
    np.savez_compressed(
        bundle_path,
        raw_u16=raw_u16,
        telemetry_u16=tele_u16,
        meta_json=json.dumps(meta),
    )

    print("Saved bundle:", bundle_path)
    print("Raw u16 min/max:", meta["raw_min"], meta["raw_max"], "Telemetry spot_c:", tele.get("spot_c"))
    return bundle_path


def save_celsius_csv(bundle_npz_path: str, csv_path: str | None = None) -> str:
    """Convert bundle raw_u16 -> Celsius CSV (120x160)."""
    z = np.load(bundle_npz_path, allow_pickle=False)
    raw_u16 = z["raw_u16"]
    meta = json.loads(str(z["meta_json"]))
    res_k = float(meta["telemetry"]["tlinear_res_k"])  # trust telemetry for the frame
    img_c = _raw_to_celsius(raw_u16, res_k)

    if csv_path is None:
        csv_path = os.path.splitext(bundle_npz_path)[0] + "_celsius.csv"

    np.savetxt(csv_path, img_c, delimiter=",", fmt="%.2f")
    print("Saved CSV:", csv_path)
    return csv_path


def save_celsius_png(bundle_npz_path: str, png_path: str | None = None, cmap="inferno") -> str:
    """Convert bundle raw_u16 -> Celsius PNG with colorbar + min/max overlay."""
    z = np.load(bundle_npz_path, allow_pickle=False)
    raw_u16 = z["raw_u16"]
    meta = json.loads(str(z["meta_json"]))
    res_k = float(meta["telemetry"]["tlinear_res_k"])
    img_c = _raw_to_celsius(raw_u16, res_k)

    if png_path is None:
        png_path = os.path.splitext(bundle_npz_path)[0] + "_celsius.png"

    # vmin, vmax = float(img_c.min()), float(img_c.max())
    vmin, vmax = np.percentile(img_c, [1, 99])	# trim errant pixels to avoid skewing temp scale

    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(img_c, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title("tCam Thermal (TLinear, °C)")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Temperature (°C)")

    ax.text(
        0.01, 0.99,
        f"Min: {vmin:.2f}°C\nMax: {vmax:.2f}°C\nSpot: {meta['telemetry'].get('spot_c', float('nan')):.2f}°C",
        transform=ax.transAxes, va="top", ha="left",
        color="white", bbox=dict(facecolor="black", alpha=0.35, edgecolor="none"),
        fontsize=10,
    )

    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved PNG:", png_path)
    return png_path


if __name__ == "__main__":
    bundle = capture_and_save_raw(outdir="data/raw", prefix="tcam", run_ffc=True, save_json=False)

    save_celsius_csv(bundle)   # <-- comment out this ONE line to stop saving CSV
    save_celsius_png(bundle)   # <-- comment out this ONE line to stop saving PNG
