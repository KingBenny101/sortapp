import os
import numpy as np
import streamlit as st
from PIL import Image

import config
from feature_extraction import get_resnet_feature_extractor, image_to_feature_vector_torch
from active_learning import init_or_load_learner, save_learner, get_prediction
from data_handler import list_unlabeled_images, load_processed_images, save_processed_images
from logging_utils import append_label_log, copy_labeled_image


def page_label_images():
    """Main page for labeling images with active learning."""
    feature_extractor, preprocess, device = get_resnet_feature_extractor()
    learner = init_or_load_learner(feature_extractor, preprocess, device)

    if "image_list" not in st.session_state:
        all_files = list_unlabeled_images(config.UNLABELED_DIR)
        processed = load_processed_images()
        remaining = [p for p in all_files if p not in processed]
        st.session_state.image_list = remaining
        st.session_state.index = 0
        st.session_state.history = []

    image_list = st.session_state.image_list
    idx = st.session_state.index

    if idx >= len(image_list):
        st.title(config.APP_TITLE)
        st.success("No more images to label in this session (all remaining are processed).")
        return

    img_path = image_list[idx]
    img = Image.open(img_path)

    # Compute features & prediction
    feat = image_to_feature_vector_torch(img, feature_extractor, preprocess, device)
    p_useful, p_useless = get_prediction(learner, feat)

    # Header row: title + uniform-font scores
    col_title, col_score = st.columns([3, 2])
    with col_title:
        st.markdown(f"### {config.APP_TITLE}")
        st.caption(f"Image {idx + 1} / {len(image_list)}  •  Remaining: {len(image_list) - idx - 1}")
    with col_score:
        st.markdown(
            f"<div style='text-align:right; font-size:1.4rem;'>"
            f"Useful: <b>{p_useful:.3f}</b><br/>"
            f"Useless: <b>{p_useless:.3f}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Options
    col_a, col_b = st.columns(2)
    auto_copy = col_a.checkbox("Only copy labeled images", value=True)
    auto_save = col_b.checkbox("Auto-save model", value=True)

    # Fixed-height image, centered via middle column
    aspect = img.width / img.height
    new_height = config.FIXED_IMAGE_HEIGHT
    new_width = int(aspect * new_height)
    img_resized = img.resize((new_width, new_height))

    left, center, right = st.columns([1, 2, 1])
    with center:
        st.image(img_resized, caption=os.path.basename(img_path), width="content")

    # Centered buttons via columns
    cols = st.columns(5)
    with cols[0]:
        btn_prev = st.button("⬅️ Back", key="btn_prev")
    with cols[1]:
        label_useful = st.button("Useful 👍", key="btn_useful")
    with cols[2]:
        label_useless = st.button("Useless 👎", key="btn_useless")
    with cols[3]:
        skip = st.button("Skip", key="btn_skip")
    with cols[4]:
        save_button = st.button("Save model", key="btn_save")

    if save_button:
        save_learner(learner)
        st.success("Model saved.")

    def apply_label(label_int: int):
        learner.teach(X=feat.reshape(1, -1), y=np.array([label_int], dtype=np.int64))
        append_label_log(img_path, label_int, p_useful)
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


def page_info():
    """Information page showing all configuration paths."""
    st.title("Info & paths")
    st.write(f"Dataset base: `{config.BASE_DATASET_DIR}`")
    st.write(f"SORTED_DIR: `{config.SORTED_DIR}`")
    st.write(f"UNLABELED_DIR: `{config.UNLABELED_DIR}`")
    st.write(f"Model path: `{config.MODEL_PATH}`")
    st.write(f"Label log: `{config.LABEL_LOG_CSV}`")
    st.write(f"Processed images CSV: `{config.PROCESSED_IMAGES_CSV}`")
    st.write(f"Labeled output base: `{config.LABELED_OUTPUT_BASE}`")