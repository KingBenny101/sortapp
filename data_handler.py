import os
import glob
from typing import List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

import config
from feature_extraction import load_and_featurize_image


def load_labeled_dataset(sorted_dir, feature_extractor, preprocess, device,
                         max_per_class: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images from sorted/useful and sorted/useless (up to max_per_class per class).
    """
    X, y = [], []
    for class_name, label in config.CLASS_TO_LABEL.items():
        class_dir = sorted_dir / class_name
        paths = sorted(glob.glob(str(class_dir / "*")))
        if max_per_class > 0:
            paths = paths[:max_per_class]
        for p in paths:
            try:
                feat = load_and_featurize_image(p, feature_extractor, preprocess, device)
                X.append(feat)
                y.append(label)
            except Exception as e:
                st.warning(f"Failed to process {p}: {e}")
    if not X:
        return np.empty((0, 2048), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(X, axis=0), np.array(y, dtype=np.int64)


def list_unlabeled_images(unlabeled_dir) -> List[str]:
    """Get list of all unlabeled images in the directory."""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = []
    for root, _, names in os.walk(unlabeled_dir):
        for n in names:
            if n.lower().endswith(exts):
                files.append(os.path.join(root, n))
    files.sort()
    return files


def load_processed_images() -> set:
    """Load set of already processed image paths."""
    if config.PROCESSED_IMAGES_CSV.exists():
        df = pd.read_csv(config.PROCESSED_IMAGES_CSV)
        return set(df["path"].tolist())
    return set()


def save_processed_images(paths: List[str]):
    """Save list of processed image paths to CSV."""
    if config.PROCESSED_IMAGES_CSV.exists():
        df = pd.read_csv(config.PROCESSED_IMAGES_CSV)
        existing = set(df["path"].tolist())
    else:
        existing = set()
        df = pd.DataFrame(columns=["path"])

    all_paths = existing.union(paths)
    out_df = pd.DataFrame(sorted(all_paths), columns=["path"])
    out_df.to_csv(config.PROCESSED_IMAGES_CSV, index=False)