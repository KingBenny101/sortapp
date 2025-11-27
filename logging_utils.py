import os
import shutil
import pandas as pd

import config


def append_label_log(img_path: str, label_int: int, proba_useful: float):
    """Append labeling action to CSV log file."""
    config.LABEL_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "path": img_path,
        "label": config.LABEL_TO_CLASS[label_int],
        "label_int": int(label_int),
        "proba_useful_before": float(proba_useful),
    }
    if config.LABEL_LOG_CSV.exists():
        df = pd.read_csv(config.LABEL_LOG_CSV)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(config.LABEL_LOG_CSV, index=False)


def copy_labeled_image(img_path: str, label_int: int):
    """
    Copy a labeled image into labeled_output/useful or labeled_output/useless,
    keeping the original file so Back/relabel works.
    """
    target_dir = config.LABELED_OUTPUT_BASE / config.LABEL_TO_CLASS[label_int]
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / os.path.basename(img_path)
    shutil.copy2(img_path, target_path)