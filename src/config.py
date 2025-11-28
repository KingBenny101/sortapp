"""Configuration settings for Sort App.

This module contains all configuration parameters including:
- Dataset and working directory paths
- Data, output, and temp directory paths
- Model and data file locations
- Class labels and mappings
- Application settings (display, auto mode, copy behavior)

Configuration is loaded from config.yaml (if values are provided) with fallback to defaults here.
"""

from pathlib import Path
import yaml

# Load config from YAML file
CONFIG_FILE = Path(__file__).parent.parent / "config.yaml"


def _load_config():
    """Load configuration from YAML file."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {CONFIG_FILE}. "
            "Please create config.yaml in the project root (next to app.py)."
        )
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_yaml_config = _load_config()


def _get_config(key, default=None):
    """Get config value from YAML, or use default if blank/not set."""
    value = _yaml_config.get(key)
    return value if value else default


# Training dataset directory (pre-labeled) - OPTIONAL
# If not provided, app runs in cold start mode
_dataset_dir = _get_config("dataset_dir")
DATASET_DIR = Path(_dataset_dir) if _dataset_dir else None

# Working directory for unlabeled images to classify - REQUIRED
_unlabeled_dir = _get_config("unlabeled_dir")
if not _unlabeled_dir:
    raise ValueError("unlabeled_dir must be provided in config.yaml")
UNLABELED_DIR = Path(_unlabeled_dir)

# Data directory (model and processed images tracking)
# Uses current working directory if not specified
DATA_DIR = Path.cwd() / "data"

# Output directory (classified images)
_output_dir_default = Path("/mnt/e/Important/Whatsapp/27112025")
OUTPUT_DIR = Path(_get_config("output_dir", str(_output_dir_default)))
OUTPUT_USEFUL_DIR = OUTPUT_DIR / "useful"
OUTPUT_USELESS_DIR = OUTPUT_DIR / "useless"

# Temp directory (cached resized images)
# Uses current working directory if not specified
TEMP_DIR = Path.cwd() / "temp"

# File paths
MODEL_PATH = DATA_DIR / "modal_resnet50_sgd.joblib"
PROCESSED_IMAGES_CSV = DATA_DIR / "processed_images.csv"

# Classes
CLASS_TO_LABEL = {"useless": 0, "useful": 1}
LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}

# Initial training limit per class from sorted dataset
MAX_PER_CLASS = _get_config("max_per_class", 2000)

# Cold start settings
ALLOW_COLD_START = _get_config(
    "allow_cold_start", True
)  # Set to True to allow starting without pre-labeled dataset
MIN_SAMPLES_PER_CLASS = _get_config(
    "min_samples_per_class", 5
)  # Minimum samples per class before model training in cold start

# Application
APP_TITLE = _get_config("app_title", "Sort App")

# Display settings
DISPLAY_WIDTH = _get_config("display_width", 480)  # 4:3 ratio width
DISPLAY_HEIGHT = _get_config("display_height", 360)  # 4:3 ratio height

# Auto mode settings
AUTO_MODE_WAIT_TIME = _get_config(
    "auto_mode_wait_time", 1.0
)  # seconds between images in auto mode
SKIP_AUTO_MODE_WAIT = _get_config(
    "skip_auto_mode_wait", True
)  # Set to True to skip wait time in auto mode
BATCH_SIZE = _get_config(
    "batch_size", 512
)  # Number of images to process in a batch for feature extraction
USE_GPU = _get_config("use_gpu", True)  # Set to True to use GPU if available (CUDA)

# Copy settings
COPY_USELESS_IMAGES = _get_config(
    "copy_useless_images", False
)  # Set to True to also copy useless images to output
