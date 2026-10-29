"""Where things live in the repo. All paths resolve from the repo root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"  # per-setup files: calibration, home pose
ASSETS_DIR = ROOT / "assets"  # SO101 URDF + meshes (not committed)
OUTPUT_DIR = ROOT / "outputs"  # debug images (not committed)

CALIB_FILE = CONFIG_DIR / "calib.json"
HOME_FILE = CONFIG_DIR / "home.json"
CLEAR_FILE = CONFIG_DIR / "clear.json"  # home with the wrist lifted clear of the base
TOOL_FILE = CONFIG_DIR / "tool.json"  # grasp-centre (TCP) offset of the gripper
WRIST_FILE = CONFIG_DIR / "wrist.json"  # wrist camera: where a centred object appears + px/mm
URDF_FILE = ASSETS_DIR / "SO101" / "so101_new_calib.urdf"


def output(name: str) -> Path:
    """Path for a debug image or log in outputs/, creating the folder if needed."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / name
