import os
import glob
import shutil
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st

import torch
import torch.nn as nn
import torchvision.models as models

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

import joblib

import config


# ------------------------------
# RESNET50 FEATURE EXTRACTOR (PyTorch)
# ------------------------------
@st.cache_resource(show_spinner=True)
def get_resnet_feature_extractor():
    """
    Load pretrained ResNet50 (torchvision) as a frozen feature extractor.[web:71][web:73][web:79][web:82]
    """
    device = torch.device("cpu")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    modules = list(model.children())[:-1]
    feature_extractor = nn.Sequential(*modules).to(device)
    feature_extractor.eval()

    weights = models.ResNet50_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    return feature_extractor, preprocess, device


def image_to_feature_vector_torch(img: Image.Image, feature_extractor, preprocess, device) -> np.ndarray:
    img = img.convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = feature_extractor(tensor)
    feat = feat.view(-1).cpu().numpy()
    return feat


def load_and_featurize_image(path, feature_extractor, preprocess, device) -> np.ndarray:
    img = Image.open(path)
    return image_to_feature_vector_torch(img, feature_extractor, preprocess, device)


# ------------------------------
# DATA LOADING
# ------------------------------
def load_labeled_dataset(sorted_dir, feature_extractor, preprocess, device,
                         max_per_class: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images from sorted/useful and sorted/useless (up to max_per_class per class).[web:71]
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
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = []
    for root, _, names in os.walk(unlabeled_dir):
        for n in names:
            if n.lower().endswith(exts):
                files.append(os.path.join(root, n))
    files.sort()
    return files


# ------------------------------
# PROCESSED IMAGES PERSISTENCE
# ------------------------------
def load_processed_images() -> set:
    if config.PROCESSED_IMAGES_CSV.exists():
        df = pd.read_csv(config.PROCESSED_IMAGES_CSV)
        return set(df["path"].tolist())
    return set()


def save_processed_images(paths: List[str]):
    if config.PROCESSED_IMAGES_CSV.exists():
        df = pd.read_csv(config.PROCESSED_IMAGES_CSV)
        existing = set(df["path"].tolist())
    else:
        existing = set()
        df = pd.DataFrame(columns=["path"])

    all_paths = existing.union(paths)
    out_df = pd.DataFrame(sorted(all_paths), columns=["path"])
    out_df.to_csv(config.PROCESSED_IMAGES_CSV, index=False)


# ------------------------------
# ACTIVE LEARNER
# ------------------------------
def create_base_estimator():
    clf = SGDClassifier(
        loss="log_loss",
        learning_rate="optimal",
        max_iter=1,      # small because we update often
        warm_start=True,
    )
    return make_pipeline(StandardScaler(), clf)


def init_or_load_learner(feature_extractor, preprocess, device):
    if config.MODEL_PATH.exists():
        learner = joblib.load(config.MODEL_PATH)
        return learner

    with st.spinner("Featurizing initial sorted dataset and training base classifier..."):
        X_train, y_train = load_labeled_dataset(
            config.SORTED_DIR,
            feature_extractor,
            preprocess,
            device,
            max_per_class=config.MAX_PER_CLASS,
        )

        if X_train.shape[0] == 0:
            raise RuntimeError("No labeled images found in 'sorted/useful' and 'sorted/useless'.")

        base_estimator = create_base_estimator()
        learner = ActiveLearner(
            estimator=base_estimator,
            query_strategy=uncertainty_sampling,
            X_training=X_train,
            y_training=y_train,
        )
    return learner


def save_learner(learner: ActiveLearner):
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(learner, config.MODEL_PATH)


# ------------------------------
# LABEL LOGGING
# ------------------------------
def append_label_log(img_path: str, label_int: int, proba_useful: float):
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


# ------------------------------
# PAGE: LABEL IMAGES
# ------------------------------
def page_label_images():
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
    feat = image_to_feature_vector_torch(img, feature_extractor, preprocess, device).reshape(1, -1)
    try:
        proba = learner.predict_proba(feat)[0]
        p_useful = float(proba[config.CLASS_TO_LABEL["useful"]])
        p_useless = float(proba[config.CLASS_TO_LABEL["useless"]])
    except Exception:
        p_useful = p_useless = 0.5

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
        learner.teach(X=feat, y=np.array([label_int], dtype=np.int64))
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


# ------------------------------
# PAGE: INFO & PATHS
# ------------------------------
def page_info():
    st.title("Info & paths")
    st.write(f"Dataset base: `{config.BASE_DATASET_DIR}`")
    st.write(f"SORTED_DIR: `{config.SORTED_DIR}`")
    st.write(f"UNLABELED_DIR: `{config.UNLABELED_DIR}`")
    st.write(f"Model path: `{config.MODEL_PATH}`")
    st.write(f"Label log: `{config.LABEL_LOG_CSV}`")
    st.write(f"Processed images CSV: `{config.PROCESSED_IMAGES_CSV}`")
    st.write(f"Labeled output base: `{config.LABELED_OUTPUT_BASE}`")


# ------------------------------
# MAIN
# ------------------------------
def main():
    st.set_page_config(page_title=config.APP_TITLE, layout="wide")

    with st.sidebar:
        page = st.radio(
            "Page",
            ["Label images", "Info & paths"],
            index=0,
        )

    if page == "Label images":
        page_label_images()
    else:
        page_info()


if __name__ == "__main__":
    main()
