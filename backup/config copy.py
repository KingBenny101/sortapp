from pathlib import Path

# Directories
SCRIPT_DIR = Path(__file__).resolve().parent

BASE_DATASET_DIR = Path("/home/kingbenny101/code/whatsapp-dataset")
SORTED_DIR = BASE_DATASET_DIR / "sorted"
UNLABELED_DIR = BASE_DATASET_DIR / "images"

# Where new labeled images are copied
LABELED_OUTPUT_BASE = SCRIPT_DIR / "labeled_output"

# Model + logs
MODEL_PATH = SCRIPT_DIR / "models" / "modal_resnet50_sgd.joblib"
LABEL_LOG_CSV = SCRIPT_DIR / "labels_log.csv"
PROCESSED_IMAGES_CSV = SCRIPT_DIR / "processed_images.csv"

# Classes
CLASS_TO_LABEL = {"useless": 0, "useful": 1}
LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}

# Initial training limit per class from /sorted
MAX_PER_CLASS = 200

# UI
APP_TITLE = "WhatsApp Image Incremental Classifier"
FIXED_IMAGE_HEIGHT = 400  # px
