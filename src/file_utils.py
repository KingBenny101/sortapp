"""File operations for managing labeled image outputs.

This module handles copying labeled images to the output directory
and managing relabeling scenarios.
"""

import os
import shutil

from . import config


def copy_labeled_image(img_path: str, label_int: int):
    """Copy a labeled image to the appropriate output directory.

    Args:
        img_path (str): Path to the source image file
        label_int (int): Integer label (0 for useless, 1 for useful)

    Behavior:
        - Useful images are always copied to output/useful/
        - Useless images are only copied if config.COPY_USELESS_IMAGES is True
        - If an image is relabeled, the old copy is removed from the previous directory
        - Original source image is preserved (copy, not move)

    Note:
        Creates output directories if they don't exist.
        Handles relabeling by removing the image from the opposite directory.
    """
    label_class = config.LABEL_TO_CLASS[label_int]
    filename = os.path.basename(img_path)

    # Determine source and target directories
    if label_class == "useful":
        target_dir = config.OUTPUT_USEFUL_DIR
        old_dir = config.OUTPUT_USELESS_DIR
    else:  # useless
        if config.COPY_USELESS_IMAGES:
            target_dir = config.OUTPUT_USELESS_DIR
            old_dir = config.OUTPUT_USEFUL_DIR
        else:
            # Need to remove from useful if it was there before
            old_path = config.OUTPUT_USEFUL_DIR / filename
            if old_path.exists():
                os.remove(old_path)
            return  # Don't copy useless images if flag is not set

    # Remove from old directory if it exists
    old_path = old_dir / filename
    if old_path.exists():
        os.remove(old_path)

    # Copy to new directory
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    shutil.copy2(img_path, target_path)
