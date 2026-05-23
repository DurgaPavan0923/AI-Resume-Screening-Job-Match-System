"""
src/explainer.py — Centroid-based feature importance explainer.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def explain_prediction(
    text: str,
    model,
    vectorizer,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """
    Return the top-N features that drove the prediction.
    """
    if not text.strip():
        return []

    try:
        vec   = vectorizer.transform([text])
        proba = model.predict_proba(vec)[0]
        
        # Find the predicted class
        top_class_idx = proba.index(max(proba))
        cls = model.classes_[top_class_idx]
        
        # Get centroid features matching the document
        centroid = model.centroids.get(cls, {})
        v = vec.rows[0]
        
        scores = []
        for word, val in v.items():
            if word in centroid and val > 0:
                scores.append((word, round(float(val * centroid[word]), 4)))
                
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    except Exception as exc:
        logger.error("Explanation failed: %s", exc)
        return []


def format_explanation(features: list[tuple[str, float]]) -> str:
    """Convert feature list to a readable bullet-point string."""
    if not features:
        return "No explanation available."
    lines = [f"• **{name}** (score: {score:.3f})" for name, score in features]
    return "\n".join(lines)