"""Where things live in the repo. All paths resolve from the repo root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"  # per-setup files: calibration, home pose
ASSETS_DIR = ROOT / "assets"  # SO101 URDF + meshes (not committed)
OUTPUT_DIR = ROOT / "outputs"  # debug images (not committed)

CALIB_FILE = CONFIG_DIR / "calib.json"
HOME_FILE = CONFIG_DIR / "home.json"
URDF_FILE = ASSETS_DIR / "SO101" / "so101_new_calib.urdf"


def output(name: str) -> Path:
    """Path for a debug image or log in outputs/, creating the folder if needed."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / name
