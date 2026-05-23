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

import os
import sys

# Add project root to path for Vercel Serverless Function imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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

# Mount static files for assets
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/assets", StaticFiles(directory=os.path.join(root_dir, "assets")), name="assets")

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


class ScoreBreakdown(BaseModel):
    similarity: float
    skills: float
    experience: float


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
    gpt_analysis:   str
    score_breakdown: ScoreBreakdown | None = None
    explanation:    str | None = None


class RewriteRequest(BaseModel):
    text: str
    mode: str = "bullet"  # "bullet", "summary", or "ats"
    role: str = "Software Engineer"


class RewriteResponse(BaseModel):
    original: str
    enhanced: str


class ChatRequest(BaseModel):
    message: str
    candidate_name: str
    candidate_score: float
    candidate_role: str
    missing_skills: list[str]
    matched_skills: list[str]


class ChatResponse(BaseModel):
    reply: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    """Serve the static web frontend dashboard at the root URL."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return FileResponse(os.path.join(root_dir, "index.html"))


@app.get("/health", response_model=HealthResponse, tags=["System"])
@app.get("/api/health", response_model=HealthResponse, tags=["System"], include_in_schema=False)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version="1.0.0")


def process_single_pdf(file_like: io.BytesIO, filename: str, job_description: str) -> AnalyzeResponse | None:
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
        return None

    import re  # noqa: PLC0415
    decision_plain = re.sub(r"<[^>]+>", "", result.decision)

    gpt_analysis = "⚠️ AI analysis unavailable (check API key / quota)."
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from src.gpt_analyzer import analyze_resume
            gpt_analysis = analyze_resume(result.raw_text, job_description)
        except Exception as exc:
            logger.error("GPT analysis failed: %s", exc)

    return AnalyzeResponse(
        name           = filename,
        score          = result.score,
        role           = result.role,
        experience     = result.experience,
        education      = result.education,
        matched_skills = list(result.skills.keys()),
        missing_skills = result.missing_skills,
        ml_roles       = [{"role": r, "confidence": c} for r, c in result.ml_roles],
        decision       = decision_plain,
        gpt_analysis   = gpt_analysis,
        score_breakdown= ScoreBreakdown(**result.score_breakdown) if result.score_breakdown else None,
        explanation    = result.explanation,
    )


@app.post("/analyze", response_model=AnalyzeResponse | list[AnalyzeResponse], tags=["Resume"])
@app.post("/api/analyze", response_model=AnalyzeResponse | list[AnalyzeResponse], tags=["Resume"], include_in_schema=False)
async def analyze(
    job_description: str  = Form(..., description="Full job description text"),
    resume:          UploadFile = File(..., description="PDF or ZIP resume file"),
) -> AnalyzeResponse | list[AnalyzeResponse]:
    """Analyse a single PDF resume or a ZIP folder of multiple PDF resumes against the supplied job description."""

    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    is_zip = resume.filename.lower().endswith(".zip") or resume.content_type in ("application/zip", "application/x-zip-compressed")

    if not is_zip and resume.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF and ZIP files are supported.")

    import zipfile  # noqa: PLC0415

    if is_zip:
        raw_zip = await resume.read()
        zip_file_like = io.BytesIO(raw_zip)
        results = []

        try:
            with zipfile.ZipFile(zip_file_like) as z:
                for file_info in z.infolist():
                    if file_info.filename.lower().endswith(".pdf") and not os.path.basename(file_info.filename).startswith("."):
                        pdf_bytes = z.read(file_info.filename)
                        pdf_file_like = io.BytesIO(pdf_bytes)
                        
                        res = process_single_pdf(
                            file_like = pdf_file_like, 
                            filename = os.path.basename(file_info.filename), 
                            job_description = job_description
                        )
                        if res:
                            results.append(res)
        except Exception as exc:
            logger.error("ZIP processing failed: %s", exc)
            raise HTTPException(status_code=400, detail=f"Failed to process ZIP archive: {exc}")

        if not results:
            raise HTTPException(status_code=422, detail="No readable PDF resumes found in the uploaded ZIP file.")
        
        return results

    else:
        raw = await resume.read()
        pdf_file_like = io.BytesIO(raw)
        res = process_single_pdf(
            file_like = pdf_file_like, 
            filename = resume.filename, 
            job_description = job_description
        )
        if res is None:
            raise HTTPException(status_code=422, detail="Could not extract text from the uploaded PDF.")
        return res


@app.post("/rewrite", response_model=RewriteResponse, tags=["AI Assistant"])
@app.post("/api/rewrite", response_model=RewriteResponse, tags=["AI Assistant"], include_in_schema=False)
async def rewrite_resume(req: RewriteRequest) -> RewriteResponse:
    original = req.text
    enhanced = ""
    
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI  # noqa: PLC0415
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            if req.mode == "bullet":
                prompt = (
                    f"You are an expert technical resume writer. Rewrite the following resume bullet point to make it more professional, "
                    f"impactful, and ATS-optimized. Use a strong action verb and quantify the result if possible. "
                    f"Target Role: {req.role}\n\nBullet Point: {req.text}"
                )
            elif req.mode == "summary":
                prompt = (
                    f"You are an expert resume writer. Generate a professional summary statement based on the following professional achievements. "
                    f"Keep it compelling, concise, and focused on key value propositions. "
                    f"Target Role: {req.role}\n\nAchievements/Context: {req.text}"
                )
            else:
                prompt = (
                    f"Optimize the following resume text to make it more ATS-friendly. Ensure keywords align with standard industry practices. "
                    f"Target Role: {req.role}\n\nText: {req.text}"
                )
                
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional resume rewriting assistant. Keep it highly relevant and technical."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3,
            )
            enhanced = response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("GPT rewrite failed: %s", exc)
            
    if not enhanced:
        # Graceful fallback enhancer
        if req.mode == "bullet":
            enhanced = f"Successfully delivered high-performance solutions for {req.role} utilizing industry best practices, improving efficiency by 15%."
        elif req.mode == "summary":
            enhanced = f"Highly motivated and result-oriented {req.role} with demonstrated expertise in delivering scalable systems and cross-functional collaboration."
        else:
            enhanced = f"Experienced {req.role} specializing in system engineering, backend operations, and agile deployment pipelines."
            
    return RewriteResponse(original=original, enhanced=enhanced)


@app.post("/chat", response_model=ChatResponse, tags=["AI Assistant"])
@app.post("/api/chat", response_model=ChatResponse, tags=["AI Assistant"], include_in_schema=False)
async def chat_assistant(req: ChatRequest) -> ChatResponse:
    reply = ""
    
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI  # noqa: PLC0415
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            prompt = (
                f"You are a helpful recruitment co-pilot assistant. Answer the recruiter's question about the candidate. "
                f"Candidate Name: {req.candidate_name}\n"
                f"Match Score: {req.candidate_score}%\n"
                f"Predicted Role: {req.candidate_role}\n"
                f"Matched Skills: {', '.join(req.matched_skills)}\n"
                f"Missing Skills: {', '.join(req.missing_skills)}\n\n"
                f"Recruiter's Question: {req.message}"
            )
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional recruitment co-pilot. Give brief, insightful, and constructive recruiter feedback."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0.4,
            )
            reply = response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("GPT chat assistant failed: %s", exc)
            
    if not reply:
        # Fallback responses
        msg = req.message.lower()
        if "missing" in msg or "skill" in msg:
            reply = f"The candidate {req.candidate_name} is missing critical keywords: {', '.join(req.missing_skills or ['None identified'])}."
        elif "why" in msg or "score" in msg or "rank" in msg:
            reply = f"{req.candidate_name} matches the target role as a {req.candidate_role} with a score of {req.candidate_score}%. The rating is influenced by their matched skills ({len(req.matched_skills)}) vs missing skills ({len(req.missing_skills)})."
        else:
            reply = f"Hello! As a recruitment co-pilot, I see {req.candidate_name} has a match score of {req.candidate_score}%. Let me know if you need specific interview questions or details on their skill gaps."
            
    return ChatResponse(reply=reply)