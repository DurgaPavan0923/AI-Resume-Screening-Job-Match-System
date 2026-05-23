"""
src/similarity.py — Compute cosine similarity between a job description
and a resume using a shared TF-IDF vectoriser.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


def compute_similarity(
    job_text: str,
    resume_text: str,
    vectorizer,
) -> float:
    """
    Return the cosine similarity (0–1) between ``job_text`` and
    ``resume_text`` using the pre-fitted ``vectorizer``.
    """
    if not job_text.strip() or not resume_text.strip():
        return 0.0

    try:
        vectors = vectorizer.transform([job_text, resume_text])
        v1 = vectors.rows[0]
        v2 = vectors.rows[1]
        
        # Calculate dot product
        dot_product = sum(v1.get(word, 0.0) * v2.get(word, 0.0) for word in v1 if word in v2)
        
        # Calculate magnitudes
        mag1 = math.sqrt(sum(val ** 2 for val in v1.values()))
        mag2 = math.sqrt(sum(val ** 2 for val in v2.values()))
        
        if mag1 == 0.0 or mag2 == 0.0:
            return 0.0
            
        score = dot_product / (mag1 * mag2)
        return max(0.0, min(1.0, float(score)))
        
    except Exception as exc:
        logger.error("Similarity computation failed: %s", exc)
        return 0.0