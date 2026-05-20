"""
api/main.py — FastAPI REST API for the AI Resume Screening System.

Run locally:
    uvicorn api.main:app --reload --port 8000

Endpoints
---------
POST /analyze   — analyse a single resume against a job description
GET  /health    — health check
"""

from __future__ import annotations

import io
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import SKILLS_PATH
from src.pipeline import run_pipeline
from src.preprocess import clean_text
from src.skill_extractor import extract_skills, load_skills
from src.train import train_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Resume Screening API",
    description="Rank and analyse resumes against a job description.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event() -> None:
    global _model, _vectorizer, _skills_db
    logger.info("Loading model…")
    _model, _vectorizer = train_model()
    _skills_db          = load_skills(SKILLS_PATH)
    logger.info("Model ready.")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    version: str


class AnalyzeResponse(BaseModel):
    name:           str
    score:          float
    role:           str
    experience:     int
    education:      list[str]
    matched_skills: list[str]
    missing_skills: list[str]
    ml_roles:       list[dict]
    decision:       str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Resume"])
async def analyze(
    job_description: str  = Form(..., description="Full job description text"),
    resume:          UploadFile = File(..., description="PDF resume file"),
) -> AnalyzeResponse:
    """Analyse a single PDF resume against the supplied job description."""

    if resume.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    # Wrap bytes in a file-like object the pipeline can consume
    raw = await resume.read()
    file_like       = io.BytesIO(raw)
    file_like.name  = resume.filename  # type: ignore[attr-defined]

    job_clean = clean_text(job_description)
    jd_skills = extract_skills(job_clean, _skills_db) or {}

    result = run_pipeline(
        file       = file_like,
        job_clean  = job_clean,
        jd_skills  = jd_skills,
        skills_db  = _skills_db,
        model      = _model,
        vectorizer = _vectorizer,
    )

    if result is None:
        raise HTTPException(
            status_code=422,
            detail="Could not extract text from the uploaded PDF.",
        )

    import re  # noqa: PLC0415
    decision_plain = re.sub(r"<[^>]+>", "", result.decision)

    return AnalyzeResponse(
        name           = result.name,
        score          = result.score,
        role           = result.role,
        experience     = result.experience,
        education      = result.education,
        matched_skills = list(result.skills.keys()),
        missing_skills = result.missing_skills,
        ml_roles       = [{"role": r, "confidence": c} for r, c in result.ml_roles],
        decision       = decision_plain,
    )