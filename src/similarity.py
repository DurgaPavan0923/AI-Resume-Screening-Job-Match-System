"""
src/similarity.py — Compute cosine similarity between a job description
and a resume using a shared TF-IDF vectoriser.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def compute_similarity(
    job_text: str,
    resume_text: str,
    vectorizer,
) -> float:
    """
    Return the cosine similarity (0–1) between ``job_text`` and
    ``resume_text`` using the pre-fitted ``vectorizer``.

    Parameters
    ----------
    job_text : str
        Cleaned job-description text.
    resume_text : str
        Cleaned resume text.
    vectorizer : fitted sklearn TfidfVectorizer
        Must already be fitted (done in ``train.py``).

    Returns
    -------
    float
        Similarity score in [0, 1].
    """
    if not job_text.strip() or not resume_text.strip():
        return 0.0

    try:
        vectors = vectorizer.transform([job_text, resume_text])
        score = cosine_similarity(vectors[0], vectors[1])[0][0]
        return float(np.clip(score, 0.0, 1.0))
    except Exception as exc:
        logger.error("Similarity computation failed: %s", exc)
        return 0.0