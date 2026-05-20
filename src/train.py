"""
src/train.py — Train (or load) the TF-IDF vectoriser and
job-role classifier used across the pipeline.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import JOB_ROLES_PATH, RANDOM_STATE, VECTORIZER_MAX_FEATURES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_training_data() -> tuple[list[str], list[str]]:
    """
    Load job-role training data from ``data/job_roles.csv``.

    Expected columns: ``text``, ``label``.
    Returns two parallel lists: texts, labels.
    """
    if not os.path.exists(JOB_ROLES_PATH):
        logger.warning(
            "job_roles.csv not found at %s — using minimal fallback data.",
            JOB_ROLES_PATH,
        )
        # Minimal fallback so the app doesn't crash on first run
        texts = [
            "python machine learning data analysis pandas numpy scikit-learn",
            "deep learning neural networks tensorflow pytorch gpu cuda",
            "django flask rest api postgresql redis celery backend",
            "react javascript typescript html css frontend ui ux",
            "aws gcp docker kubernetes devops ci cd terraform",
            "data visualisation tableau power bi sql reporting analyst",
        ]
        labels = [
            "Data Scientist",
            "AI Engineer",
            "Backend Developer",
            "Frontend Developer",
            "DevOps Engineer",
            "Data Analyst",
        ]
        return texts, labels

    df = pd.read_csv(JOB_ROLES_PATH)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"job_roles.csv must contain columns {required}; "
            f"found {set(df.columns)}"
        )
    df = df.dropna(subset=["text", "label"])
    return df["text"].tolist(), df["label"].tolist()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_model() -> tuple:
    """
    Build and fit a TF-IDF vectoriser + Logistic Regression classifier.

    Returns
    -------
    tuple[LogisticRegression, TfidfVectorizer]
        ``(model, vectorizer)`` — both already fitted.
    """
    texts, labels = _load_training_data()

    vectorizer = TfidfVectorizer(
        max_features=VECTORIZER_MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b",
    )

    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(
        max_iter=1_000,
        random_state=RANDOM_STATE,
        multi_class="multinomial",
        solver="lbfgs",
        C=5.0,
    )
    model.fit(X, labels)

    logger.info(
        "Model trained on %d samples with %d features.",
        len(texts),
        X.shape[1],
    )
    return model, vectorizer