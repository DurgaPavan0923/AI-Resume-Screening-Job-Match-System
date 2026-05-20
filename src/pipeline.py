"""
src/pipeline.py — End-to-end processing pipeline for a single resume.

Orchestrates: parse → clean → extract → score → predict → explain.
Call ``run_pipeline()`` from app.py or the API layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config import MAX_EXPERIENCE_YEARS, WEIGHT_EXPERIENCE, WEIGHT_SIMILARITY, WEIGHT_SKILL
from src.education_parser import extract_education
from src.experience_extractor import extract_experience
from src.job_predictor import predict_roles
from src.pdf_parser import parse_pdf
from src.preprocess import clean_text
from src.similarity import compute_similarity
from src.skill_extractor import extract_skills

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class CandidateResult:
    name:           str
    raw_text:       str
    clean_text:     str
    score:          float                        # 0–100
    skills:         dict[str, int]  = field(default_factory=dict)
    missing_skills: list[str]       = field(default_factory=list)
    role:           str             = ""
    ml_roles:       list[tuple]     = field(default_factory=list)
    experience:     int             = 0
    education:      list[str]       = field(default_factory=list)
    gpt_analysis:   str             = ""
    decision:       str             = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_pipeline(
    file,
    job_clean: str,
    jd_skills: dict[str, int],
    skills_db: dict[str, int],
    model,
    vectorizer,
) -> CandidateResult | None:
    """
    Process a single uploaded resume file and return a ``CandidateResult``.

    Returns ``None`` if text extraction fails.
    """
    try:
        raw_text = parse_pdf(file)
        if not raw_text.strip():
            logger.warning("Empty text extracted from %s", getattr(file, "name", "?"))
            return None

        clean = clean_text(raw_text)

        sim_score  = compute_similarity(job_clean, clean, vectorizer)
        skills     = extract_skills(clean, skills_db) or {}

        # Weighted skill score
        if jd_skills:
            total   = sum(skills_db.get(s, 1) for s in jd_skills)
            matched = sum(skills_db.get(s, 1) for s in skills if s in jd_skills)
            s_score = matched / total if total else 0.0
        else:
            s_score = 0.0

        experience = extract_experience(clean)
        education  = extract_education(clean)
        roles, ml_roles = predict_roles(clean, skills, model, vectorizer)

        final_score = (
            WEIGHT_SIMILARITY * sim_score
            + WEIGHT_SKILL * s_score
            + WEIGHT_EXPERIENCE * min(experience / MAX_EXPERIENCE_YEARS, 1.0)
        )
        score_pct = round(final_score * 100, 2)

        # Skill gap
        jd_set     = set(jd_skills.keys())
        resume_set = set(skills.keys())
        missing    = sorted(jd_set - resume_set)

        # Decision badge (HTML)
        if score_pct >= 75:
            decision = "<span style='color:#1de9b6;font-weight:700'>🟢 Hire</span>"
        elif score_pct >= 50:
            decision = "<span style='color:#ffb300;font-weight:700'>🟡 Consider</span>"
        else:
            decision = "<span style='color:#ff5252;font-weight:700'>🔴 Reject</span>"

        return CandidateResult(
            name           = getattr(file, "name", "unknown"),
            raw_text       = raw_text,
            clean_text     = clean,
            score          = score_pct,
            skills         = skills,
            missing_skills = missing,
            role           = ", ".join(roles),
            ml_roles       = ml_roles,
            experience     = experience,
            education      = education,
            decision       = decision,
        )

    except Exception as exc:
        logger.error("Pipeline error for %s: %s", getattr(file, "name", "?"), exc)
        return None