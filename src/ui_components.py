"""Streamlit UI components for the image classifier.

This module contains all UI page functions and helper utilities for:
- Main labeling interface with manual and auto modes
- Settings configuration page
- Information and paths display page
- Image resizing and caching
- Temporary directory cleanup
"""

import os
import hashlib
import numpy as np
import streamlit as st
from PIL import Image

from . import config
from .feature_extraction import (
    get_resnet_feature_extractor,
    image_to_feature_vector_torch,
    batch_extract_features,
)
from .active_learning import (
    init_or_load_learner,
    save_learner,
    get_prediction,
    batch_teach,
    create_learner_from_bootstrap,
    check_can_bootstrap,
)
from .data_handler import (
    list_unlabeled_images,
    load_processed_images,
    save_processed_images,
)
from .file_utils import copy_labeled_image


def cleanup_temp_directory():
    """Remove all cached resized images from the temp directory.

    Note:
        Deletes and recreates the entire temp directory.
        Called once per session on first app load.
    """
    if config.TEMP_DIR.exists():
        import shutil

        shutil.rmtree(config.TEMP_DIR)
    config.TEMP_DIR.mkdir(exist_ok=True)


def resize_image_to_display(img_path, target_width=480, target_height=360):
    """Resize an image to fixed dimensions and cache the result.

    Args:
        img_path (str): Path to the source image file
        target_width (int, optional): Target width in pixels. Defaults to 480.
        target_height (int, optional): Target height in pixels. Defaults to 360.

    Returns:
        str: Path to the cached resized image

    Note:
        Uses MD5 hash of path for cache filename to avoid name collisions.
        Returns cached version if it already exists.
        Images are resized to exact dimensions (may change aspect ratio).
        Saves resized images as JPEG with 85% quality.
    """
    # Create temp directory if it doesn't exist
    config.TEMP_DIR.mkdir(exist_ok=True)

    # Generate cache filename based on original path and target size
    path_hash = hashlib.md5(str(img_path).encode()).hexdigest()[:8]
    cache_filename = f"{path_hash}_{target_width}x{target_height}.jpg"
    cache_path = config.TEMP_DIR / cache_filename

    # Return cached version if it exists
    if cache_path.exists():
        return str(cache_path)

    # Open and resize image
    img = Image.open(img_path)

    # Convert to RGB if necessary
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize to exact dimensions (may distort aspect ratio)
    img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

    # Save to cache
    img_resized.save(cache_path, "JPEG", quality=85)

    return str(cache_path)


def process_batch_auto(
    image_paths, learner, feature_extractor, preprocess, device, auto_copy=True
):
    """Process a batch of images in auto mode with batch feature extraction and model updates.

    Args:
        image_paths (List[str]): List of image paths to process
        learner (ActiveLearner): The active learning model
        feature_extractor: ResNet50 feature extractor
        preprocess: Image preprocessing transforms
        device: Torch device (CPU/GPU)
        auto_copy (bool): Whether to copy labeled images to output

    Returns:
        int: Number of images successfully processed

    Note:
        Significantly faster than processing one-by-one:
        - Batch feature extraction on GPU
        - Single model update for all samples
        - Deferred file operations
    """
    if not image_paths:
        return 0

    # Batch extract features
    features = batch_extract_features(
        image_paths, feature_extractor, preprocess, device
    )

    if len(features) == 0:
        return 0

    # Get predictions for all images
    try:
        probas = learner.predict_proba(features)
    except (AttributeError, ValueError):
        # If model not ready, skip batch
        return 0

    # Determine labels based on probabilities
    labels = []
    processed_paths = []

    for img_path, proba in zip(image_paths, probas):
        p_useful = float(proba[config.CLASS_TO_LABEL["useful"]])
        p_useless = float(proba[config.CLASS_TO_LABEL["useless"]])

        # Auto-label based on highest probability
        label_int = (
            config.CLASS_TO_LABEL["useful"]
            if p_useful > p_useless
            else config.CLASS_TO_LABEL["useless"]
        )

        labels.append(label_int)
        processed_paths.append(img_path)

        # Copy to output if enabled
        if auto_copy:
            copy_labeled_image(img_path, label_int)

    # Batch teach the model
    if len(labels) > 0:
        batch_teach(learner, features, np.array(labels, dtype=np.int64))

    # Save processed images list
    save_processed_images(processed_paths)

    return len(processed_paths)


def page_label_images():
    """Main page for labeling images with incremental learning.

    Features:
        - Displays unlabeled images one at a time with model predictions
        - Allows manual labeling (Useful/Useless/Skip)
        - Supports auto mode for automatic classification based on model predictions
        - Provides back button to relabel previous images
        - Shows real-time progress with image count and progress bar in auto mode
        - Incrementally updates the model with each new label

    Session State:
        - temp_cleaned (bool): Whether temp directory has been cleaned this session
        - image_list (List[str]): List of unlabeled image paths to process
        - index (int): Current position in the image list
        - history (List[dict]): History of labeling actions for undo support
        - auto_mode (bool): Whether auto mode is currently active

    Note:
        Auto mode automatically labels images based on highest prediction probability
        and advances every AUTO_MODE_WAIT_TIME seconds.
    """
    # Clean temp directory on first load
    if "temp_cleaned" not in st.session_state:
        cleanup_temp_directory()
        st.session_state.temp_cleaned = True

    feature_extractor, preprocess, device = get_resnet_feature_extractor()
    learner = init_or_load_learner(feature_extractor, preprocess, device)

    # Initialize cold start tracking in session state
    if "bootstrap_features" not in st.session_state:
        st.session_state.bootstrap_features = []
        st.session_state.bootstrap_labels = []
        st.session_state.bootstrap_paths = []

    if "image_list" not in st.session_state:
        all_files = list_unlabeled_images(config.UNLABELED_DIR)
        processed = load_processed_images()
        remaining = [p for p in all_files if p not in processed]
        st.session_state.image_list = remaining
        st.session_state.index = 0
        st.session_state.history = []
        st.session_state.auto_mode = False

    image_list = st.session_state.image_list
    idx = st.session_state.index

    if idx >= len(image_list):
        st.title(config.APP_TITLE)
        st.success(
            "No more images to label in this session (all remaining are processed)."
        )
        return

    img_path = image_list[idx]

    # Load original image for feature extraction
    original_img = Image.open(img_path)

    # Create resized version for display
    display_img_path = resize_image_to_display(
        img_path, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT
    )

    # Compute features & prediction on original image
    feat = image_to_feature_vector_torch(
        original_img, feature_extractor, preprocess, device
    )

    # Get prediction (will be 0.5/0.5 if learner is None or not trained)
    if learner is not None:
        p_useful, p_useless = get_prediction(learner, feat)
    else:
        p_useful, p_useless = 0.5, 0.5

    # Centered title
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.title(config.APP_TITLE)

    # Cold start status message
    if learner is None:
        _, col2, _ = st.columns([1, 2, 1])
        with col2:
            bootstrap_labels_array = np.array(st.session_state.bootstrap_labels)
            if len(bootstrap_labels_array) > 0:
                unique, counts = np.unique(bootstrap_labels_array, return_counts=True)
                label_counts = dict(zip(unique, counts))
                useful_count = label_counts.get(config.CLASS_TO_LABEL["useful"], 0)
                useless_count = label_counts.get(config.CLASS_TO_LABEL["useless"], 0)
                
                st.warning(
                    f"🔄 **Bootstrap Mode**: Collecting initial samples...\n\n"
                    f"Useful: {useful_count}/{config.MIN_SAMPLES_PER_CLASS} | "
                    f"Useless: {useless_count}/{config.MIN_SAMPLES_PER_CLASS}"
                )
            else:
                st.info(
                    f"🚀 **Cold Start**: Label at least {config.MIN_SAMPLES_PER_CLASS} "
                    "images per class to train the model."
                )

    # Model prediction scores centered
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        metric_cols = st.columns(2)
        with metric_cols[0]:
            st.metric("Useful Score", f"{p_useful:.3f}")
        with metric_cols[1]:
            st.metric("Useless Score", f"{p_useless:.3f}")

    # Get settings from session state
    auto_copy = st.session_state.get("auto_copy", True)
    auto_save = st.session_state.get("auto_save", True)

    # Filename with image count - centered
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.info(f"**{os.path.basename(img_path)}** ({idx + 1} of {len(image_list)})")

    # Image display with container
    with st.container():
        _, col2, _ = st.columns([1, 2, 1])
        with col2:
            st.image(
                display_img_path, width=config.DISPLAY_WIDTH, use_container_width=False
            )

    # Button layout - centered
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        # Check if auto mode is active
        auto_mode_active = st.session_state.get("auto_mode", False)

        # First row: Useful and Useless buttons
        button_row1 = st.columns(2)
        with button_row1[0]:
            label_useful = st.button(
                "✅ Useful",
                key="btn_useful",
                use_container_width=True,
                disabled=auto_mode_active,
            )
        with button_row1[1]:
            label_useless = st.button(
                "❌ Useless",
                key="btn_useless",
                use_container_width=True,
                disabled=auto_mode_active,
            )

        # Second row: Back and Skip buttons
        button_row2 = st.columns(2)
        with button_row2[0]:
            btn_prev = st.button(
                "⬅️ Back",
                key="btn_prev",
                use_container_width=True,
                disabled=auto_mode_active,
            )
        with button_row2[1]:
            skip = st.button(
                "⏭️ Skip",
                key="btn_skip",
                use_container_width=True,
                disabled=auto_mode_active,
            )

        # Third row: Auto Mode and Save Model buttons
        button_row3 = st.columns(2)
        with button_row3[0]:
            if auto_mode_active:
                stop_auto = st.button(
                    "⏸️ Stop Auto", key="btn_stop_auto", use_container_width=True
                )
                if stop_auto:
                    st.session_state.auto_mode = False
                    st.rerun()
            else:
                auto_button = st.button(
                    "▶️ Auto Mode",
                    key="btn_auto",
                    use_container_width=True,
                    disabled=(learner is None),  # Disable in cold start
                )
                if auto_button:
                    st.session_state.auto_mode = True
                    st.rerun()
        with button_row3[1]:
            save_button = st.button(
                "💾 Save Model",
                key="btn_save",
                use_container_width=True,
                disabled=auto_mode_active or (learner is None),  # Disable if no model
            )

    if save_button and learner is not None:
        save_learner(learner)
        st.success("Model saved.")

    def apply_label(label_int: int):
        nonlocal learner
        
        # Cold start mode: collect bootstrap samples
        if learner is None:
            st.session_state.bootstrap_features.append(feat)
            st.session_state.bootstrap_labels.append(label_int)
            st.session_state.bootstrap_paths.append(img_path)
            
            # Check if we can bootstrap the model
            bootstrap_labels_array = np.array(st.session_state.bootstrap_labels)
            if check_can_bootstrap(bootstrap_labels_array):
                # Create model from bootstrap samples
                bootstrap_features_array = np.stack(st.session_state.bootstrap_features)
                learner = create_learner_from_bootstrap(
                    bootstrap_features_array, bootstrap_labels_array
                )
                save_learner(learner)
                
                # Save all bootstrap paths as processed
                save_processed_images(st.session_state.bootstrap_paths)
                
                # Copy bootstrap images if enabled
                if auto_copy:
                    for path, label in zip(
                        st.session_state.bootstrap_paths, st.session_state.bootstrap_labels
                    ):
                        copy_labeled_image(path, label)
                
                st.success(
                    f"🎉 Model initialized with {len(bootstrap_labels_array)} samples! "
                    "Now using incremental learning."
                )
            else:
                # Just save as processed for now
                save_processed_images([img_path])
                if auto_copy:
                    copy_labeled_image(img_path, label_int)
        else:
            # Normal incremental learning
            learner.teach(X=feat.reshape(1, -1), y=np.array([label_int], dtype=np.int64))
            if auto_copy:
                copy_labeled_image(img_path, label_int)
            save_processed_images([img_path])
            if auto_save:
                save_learner(learner)
        
        st.session_state.history.append(
            {"path": img_path, "label_int": label_int, "proba": p_useful}
        )

    if btn_prev and st.session_state.history:
        st.session_state.index = max(0, st.session_state.index - 1)
        st.session_state.history.pop()
        st.rerun()

    if label_useful:
        apply_label(config.CLASS_TO_LABEL["useful"])
        st.session_state.index += 1
        st.rerun()
    elif label_useless:
        apply_label(config.CLASS_TO_LABEL["useless"])
        st.session_state.index += 1
        st.rerun()
    elif skip:
        st.session_state.index += 1
        st.rerun()

    # Auto mode processing
    if st.session_state.get("auto_mode", False):
        # Get remaining images for batch processing
        remaining_images = image_list[idx:]
        batch_size = min(config.BATCH_SIZE, len(remaining_images))

        if batch_size > 0:
            # Show progress
            _, col2, _ = st.columns([1, 2, 1])
            with col2:
                st.info(f"🚀 Processing batch of {batch_size} images...")
                progress_bar = st.progress(0)

            # Process batch
            batch_paths = remaining_images[:batch_size]
            processed_count = process_batch_auto(
                batch_paths, learner, feature_extractor, preprocess, device, auto_copy
            )

            # Update progress
            progress_bar.progress(1.0)

            # Save model after batch
            if auto_save:
                save_learner(learner)

            # Update index and rerun
            st.session_state.index += processed_count

            # Show completion message
            with col2:
                st.success(f"✅ Processed {processed_count} images")
                if st.session_state.index >= len(image_list):
                    st.balloons()
                    st.session_state.auto_mode = False

        st.rerun()


def page_settings():
    """Settings page for configuring application behavior.

    Available Settings:
        - Copy labeled images: Whether to copy labeled images to output folder
        - Auto-save model: Whether to save model after each labeling action

    Session State:
        - auto_copy (bool): If True, copies labeled images to output directory
        - auto_save (bool): If True, saves model after each label

    Note:
        Settings are persisted in Streamlit session state and apply immediately.
    """
    st.title("Settings")

    st.subheader("Labeling Options")

    # Copy labeled images setting
    auto_copy = st.checkbox(
        "Copy labeled images to output folder",
        value=st.session_state.get("auto_copy", True),
        key="auto_copy",
        help="When enabled, labeled images are copied to labeled_output/useful or labeled_output/useless",
    )

    # Auto-save model setting
    auto_save = st.checkbox(
        "Auto-save model after each label",
        value=st.session_state.get("auto_save", True),
        key="auto_save",
        help="When enabled, the model is automatically saved after each labeling action",
    )

    st.subheader("Current Settings")
    st.write(f"- Copy labeled images: **{'Enabled' if auto_copy else 'Disabled'}**")
    st.write(f"- Auto-save model: **{'Enabled' if auto_save else 'Disabled'}**")


def page_info():
    """Information page displaying all configuration paths and settings.

    Displays:
        - Training dataset directory path
        - Unlabeled images directory path
        - Output directory path
        - Model file path
        - Processed images CSV path

    Note:
        All paths are read from the config module and displayed for reference.
    """
    st.title("Info")
    st.write(f"Training dataset: `{config.DATASET_DIR}`")
    st.write(f"Unlabeled images: `{config.UNLABELED_DIR}`")
    st.write(f"Output directory: `{config.OUTPUT_DIR}`")
    st.write(f"Model path: `{config.MODEL_PATH}`")
    st.write(f"Processed images CSV: `{config.PROCESSED_IMAGES_CSV}`")
