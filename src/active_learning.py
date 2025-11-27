"""Active learning model management and predictions.

This module handles:
- Creating and initializing the incremental learning classifier
- Loading and saving model state
- Making predictions with probability scores
- Training the model with new labeled examples
"""

import streamlit as st
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

from . import config
from .data_handler import load_labeled_dataset


def create_base_estimator():
    """Create an SGD classifier pipeline with feature scaling.

    Returns:
        Pipeline: sklearn pipeline containing StandardScaler and SGDClassifier

    Note:
        Uses log_loss for probabilistic predictions and warm_start for incremental learning.
        The classifier is configured for online learning with max_iter=1.
    """
    clf = SGDClassifier(
        loss="log_loss",
        learning_rate="optimal",
        max_iter=1,  # small because we update often
        warm_start=True,
    )
    return make_pipeline(StandardScaler(), clf)


def init_or_load_learner(feature_extractor, preprocess, device):
    """Initialize a new active learner or load existing one from disk.

    Args:
        feature_extractor (nn.Sequential): Pre-trained ResNet50 feature extractor
        preprocess (transforms.Compose): Image preprocessing transforms
        device (torch.device): Device to run inference on

    Returns:
        ActiveLearner: Initialized or loaded active learning model

    Raises:
        RuntimeError: If no labeled images are found in the training dataset

    Note:
        If a saved model exists, it is loaded from disk.
        Otherwise, creates a new model trained on the sorted dataset.
    """
    if config.MODEL_PATH.exists():
        learner = joblib.load(config.MODEL_PATH)
        return learner

    with st.spinner(
        "Featurizing initial sorted dataset and training base classifier..."
    ):
        X_train, y_train = load_labeled_dataset(
            config.DATASET_DIR,
            feature_extractor,
            preprocess,
            device,
            max_per_class=config.MAX_PER_CLASS,
        )

        if X_train.shape[0] == 0:
            raise RuntimeError(
                "No labeled images found in 'sorted/useful' and 'sorted/useless'."
            )

        base_estimator = create_base_estimator()
        learner = ActiveLearner(
            estimator=base_estimator,
            query_strategy=uncertainty_sampling,
            X_training=X_train,
            y_training=y_train,
        )
    return learner


def save_learner(learner: ActiveLearner):
    """Persist the active learner model to disk.

    Args:
        learner (ActiveLearner): The active learning model to save

    Note:
        Creates the data directory if it doesn't exist.
        Saves to the path specified in config.MODEL_PATH.
    """
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(learner, config.MODEL_PATH)


def get_prediction(learner: ActiveLearner, feat: np.ndarray) -> tuple[float, float]:
    """Get model prediction probabilities for an image feature vector.

    Args:
        learner (ActiveLearner): The active learning model
        feat (np.ndarray): 2048-dimensional feature vector from ResNet50

    Returns:
        tuple[float, float]: A tuple containing:
            - p_useful (float): Probability that the image is useful (0.0-1.0)
            - p_useless (float): Probability that the image is useless (0.0-1.0)

    Note:
        Returns (0.5, 0.5) if prediction fails (e.g., model not trained yet).
    """
    try:
        proba = learner.predict_proba(feat.reshape(1, -1))[0]
        p_useful = float(proba[config.CLASS_TO_LABEL["useful"]])
        p_useless = float(proba[config.CLASS_TO_LABEL["useless"]])
        return p_useful, p_useless
    except (AttributeError, ValueError, IndexError):
        # AttributeError: Learner not properly initialized
        # ValueError: Invalid feature shape or unfitted estimator
        # IndexError: Class label index out of bounds
        return 0.5, 0.5
