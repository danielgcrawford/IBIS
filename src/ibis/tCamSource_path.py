from pathlib import Path
import sys

def add_tcam_to_path():
    root = Path(__file__).resolve().parents[2]  # .../IBIS
    tcam_py = root / "tCamSource" / "tCam" / "python"
    if not tcam_py.exists():
        raise FileNotFoundError(f"tCam python folder not found at: {tcam_py}")
    sys.path.insert(0, str(tcam_py))
