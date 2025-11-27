"""Dataset handling and image processing utilities.

This module provides functions for:
- Loading pre-labeled training data from the sorted dataset
- Listing unlabeled images for classification
- Tracking processed images via CSV persistence
"""

import os
import glob
from typing import List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from . import config
from .feature_extraction import load_and_featurize_image


def load_labeled_dataset(
    sorted_dir, feature_extractor, preprocess, device, max_per_class: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and featurize pre-labeled images from the training dataset.

    Args:
        sorted_dir (Path): Path to directory containing 'useful' and 'useless' subdirectories
        feature_extractor (nn.Sequential): Pre-trained ResNet50 feature extractor
        preprocess (transforms.Compose): Image preprocessing transforms
        device (torch.device): Device to run inference on
        max_per_class (int): Maximum number of images to load per class

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X (np.ndarray): Feature matrix of shape (n_samples, 2048)
            - y (np.ndarray): Label array of shape (n_samples,) with integer class labels

    Note:
        Displays warnings for images that fail to process.
        Returns empty arrays if no valid images are found.
    """
    X, y = [], []
    for class_name, label in config.CLASS_TO_LABEL.items():
        class_dir = sorted_dir / class_name
        paths = sorted(glob.glob(str(class_dir / "*")))
        if max_per_class > 0:
            paths = paths[:max_per_class]
        for p in paths:
            try:
                feat = load_and_featurize_image(
                    p, feature_extractor, preprocess, device
                )
                X.append(feat)
                y.append(label)
            except (OSError, ValueError, RuntimeError) as e:
                # OSError: File I/O errors (missing, corrupt, permissions)
                # ValueError: Invalid image data or format
                # RuntimeError: Model/tensor errors during feature extraction
                st.warning(f"Failed to process {p}: {e}")
    if not X:
        return np.empty((0, 2048), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(X, axis=0), np.array(y, dtype=np.int64)


def list_unlabeled_images(unlabeled_dir) -> List[str]:
    """Recursively find all image files in the unlabeled directory.

    Args:
        unlabeled_dir (Path): Directory to search for unlabeled images

    Returns:
        List[str]: Sorted list of absolute paths to image files

    Note:
        Supports .jpg, .jpeg, .png, .bmp, and .webp formats.
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = []
    for root, _, names in os.walk(unlabeled_dir):
        for n in names:
            if n.lower().endswith(exts):
                files.append(os.path.join(root, n))
    files.sort()
    return files


def load_processed_images() -> set:
    """Load the set of already processed image paths from CSV.

    Returns:
        set: Set of absolute paths to images that have been processed

    Note:
        Returns empty set if the processed images CSV doesn't exist yet.
    """
    if config.PROCESSED_IMAGES_CSV.exists():
        df = pd.read_csv(config.PROCESSED_IMAGES_CSV)
        return set(df["path"].tolist())
    return set()


def save_processed_images(paths: List[str]):
    """Add newly processed image paths to the CSV tracking file.

    Args:
        paths (List[str]): List of image paths to mark as processed

    Note:
        Merges with existing processed images and saves sorted results.
        Creates the data directory if it doesn't exist.
    """
    if config.PROCESSED_IMAGES_CSV.exists():
        df = pd.read_csv(config.PROCESSED_IMAGES_CSV)
        existing = set(df["path"].tolist())
    else:
        existing = set()
        df = pd.DataFrame(columns=["path"])

    all_paths = existing.union(paths)
    config.PROCESSED_IMAGES_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(sorted(all_paths), columns=["path"])
    out_df.to_csv(config.PROCESSED_IMAGES_CSV, index=False)
