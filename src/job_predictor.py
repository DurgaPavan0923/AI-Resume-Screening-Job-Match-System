"""
src/job_predictor.py — Predict likely job roles from resume text.
"""

from __future__ import annotations

import logging

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

        # Sort classes descending by probability
        ranked = sorted(
            list(zip(classes, proba)),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        role_names = [r for r, p in ranked]
        ml_roles   = [(r, round(float(p) * 100, 1)) for r, p in ranked]
        
        return role_names, ml_roles

    except Exception as exc:
        logger.error("Role prediction failed: %s", exc)
        return [], []