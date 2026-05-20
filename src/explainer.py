"""
src/explainer.py — SHAP / LIME-based feature importance explainer
for the job-role classifier.

Generates human-readable explanations of why a particular role
was predicted for a candidate.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def explain_prediction(
    text: str,
    model,
    vectorizer,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """
    Return the top-N TF-IDF features that drove the model's
    top prediction, using the model's own coefficients.

    Parameters
    ----------
    text : str
        Cleaned resume/JD text.
    model : fitted LogisticRegression
    vectorizer : fitted TfidfVectorizer
    top_n : int
        Number of features to return.

    Returns
    -------
    list[tuple[str, float]]
        ``[(feature_name, importance_score), ...]`` sorted descending.
    """
    if not text.strip():
        return []

    try:
        vec   = vectorizer.transform([text])
        proba = model.predict_proba(vec)[0]
        top_class_idx = int(np.argmax(proba))

        # Coefficients for the predicted class
        coefs = model.coef_[top_class_idx]
        feature_names = vectorizer.get_feature_names_out()

        # Weight coefficients by the actual TF-IDF value in this document
        tfidf_vals = vec.toarray()[0]
        scores     = coefs * tfidf_vals

        # Top-N positive contributors
        ranked = np.argsort(scores)[::-1][:top_n]
        return [
            (feature_names[i], round(float(scores[i]), 4))
            for i in ranked
            if scores[i] > 0
        ]

    except Exception as exc:
        logger.error("Explanation failed: %s", exc)
        return []


def format_explanation(features: list[tuple[str, float]]) -> str:
    """Convert feature list to a readable bullet-point string."""
    if not features:
        return "No explanation available."
    lines = [f"• **{name}** (score: {score:.3f})" for name, score in features]
    return "\n".join(lines)