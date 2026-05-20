"""
src/job_predictor.py — Predict likely job roles from resume text
using the trained classifier.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Number of top roles to return
TOP_N = 3


def predict_roles(
    clean_text: str,
    skills: dict,
    model,
    vectorizer,
    top_n: int = TOP_N,
) -> tuple[list[str], list[tuple[str, float]]]:
    """
    Predict the most likely job roles for a candidate.

    Parameters
    ----------
    clean_text : str
        Pre-processed resume text.
    skills : dict
        Extracted skill → weight mapping (used to augment text).
    model : fitted LogisticRegression
    vectorizer : fitted TfidfVectorizer
    top_n : int
        How many roles to return.

    Returns
    -------
    tuple[list[str], list[tuple[str, float]]]
        ``(role_names, [(role, confidence_pct), ...])``
        where confidence is rounded to 1 decimal place.
    """
    if not clean_text.strip():
        return [], []

    # Augment with skill names (repeat once to boost their TF weight)
    skill_boost = " ".join(skills.keys()) if skills else ""
    augmented   = f"{clean_text} {skill_boost}".strip()

    try:
        vec   = vectorizer.transform([augmented])
        proba = model.predict_proba(vec)[0]
        classes = model.classes_

        # Sort descending by confidence
        ranked_idx = np.argsort(proba)[::-1][:top_n]
        role_names = [classes[i] for i in ranked_idx]
        ml_roles   = [
            (classes[i], round(float(proba[i]) * 100, 1))
            for i in ranked_idx
        ]
        return role_names, ml_roles

    except Exception as exc:
        logger.error("Role prediction failed: %s", exc)
        return [], []