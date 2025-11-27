import streamlit as st
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

import config
from data_handler import load_labeled_dataset


def create_base_estimator():
    """Create base SGD classifier with incremental learning support."""
    clf = SGDClassifier(
        loss="log_loss",
        learning_rate="optimal",
        max_iter=1,      # small because we update often
        warm_start=True,
    )
    return make_pipeline(StandardScaler(), clf)


def init_or_load_learner(feature_extractor, preprocess, device):
    """Initialize new learner or load existing one from disk."""
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
    """Save active learner model to disk."""
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(learner, config.MODEL_PATH)


def get_prediction(learner: ActiveLearner, feat: np.ndarray) -> tuple[float, float]:
    """Get model prediction probabilities for useful/useless classes."""
    try:
        proba = learner.predict_proba(feat.reshape(1, -1))[0]
        p_useful = float(proba[config.CLASS_TO_LABEL["useful"]])
        p_useless = float(proba[config.CLASS_TO_LABEL["useless"]])
        return p_useful, p_useless
    except Exception:
        return 0.5, 0.5