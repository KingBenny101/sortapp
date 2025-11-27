"""Configuration settings for Sort App.

This module contains all configuration parameters including:
- Dataset and working directory paths
- Data, output, and temp directory paths
- Model and data file locations
- Class labels and mappings
- Application settings (display, auto mode, copy behavior)
"""

from pathlib import Path

# Training dataset directory (pre-labeled)
DATASET_DIR = Path("/home/kingbenny101/code/whatsapp-dataset/sorted")

# Working directory for unlabeled images to classify
UNLABELED_DIR = Path("/home/kingbenny101/code/whatsapp-dataset/images")

# Data directory (model and processed images tracking)
# Uses current working directory if not specified
DATA_DIR = Path.cwd() / "data"

# Output directory (classified images)
# Uses current working directory if not specified
OUTPUT_DIR = Path.cwd() / "output"
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
MAX_PER_CLASS = 200

# Cold start settings
ALLOW_COLD_START = True  # Set to True to allow starting without pre-labeled dataset
MIN_SAMPLES_PER_CLASS = (
    5  # Minimum samples per class before model training in cold start
)

# Application
APP_TITLE = "Sort App"

# Display settings
DISPLAY_WIDTH = 480  # 4:3 ratio width
DISPLAY_HEIGHT = 360  # 4:3 ratio height

# Auto mode settings
AUTO_MODE_WAIT_TIME = 1.0  # seconds between images in auto mode
SKIP_AUTO_MODE_WAIT = True  # Set to True to skip wait time in auto mode
BATCH_SIZE = 320  # Number of images to process in a batch for feature extraction
USE_GPU = True  # Set to True to use GPU if available (CUDA)

# Copy settings
COPY_USELESS_IMAGES = False  # Set to True to also copy useless images to output
