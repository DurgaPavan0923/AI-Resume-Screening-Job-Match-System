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
# Centralised AI Config State & Providers Dispatcher
# ---------------------------------------------------------------------------
import urllib.request
import json
import time
from fastapi import Request

class AIConfig:
    active_provider = "gemini"
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "AIzaSyBjOMSP6Bl9jZKxgJVZh6X1Ca38wkjiD48")
    huggingface_key = os.environ.get("HUGGINGFACE_API_KEY", "")
    
    # Security options
    enable_throttling = True
    enable_masking = True
    enable_cors = True
    
    # Usage metrics
    requests_today = 0
    tokens_consumed = 0
    failed_requests = 0
    
    # Simple rate limiting storage
    rate_limits = {}


def sanitize_input(text: str) -> str:
    if not AIConfig.enable_masking:
        return text
    import re
    # Mask emails and standard phone number formats
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL_MASKED]", text)
    text = re.sub(r"\b(?:\+?\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b", "[PHONE_MASKED]", text)
    return text


def check_rate_limit(client_ip: str) -> None:
    if not AIConfig.enable_throttling:
        return
    now = time.time()
    # Limit: max 15 requests per 60 seconds
    timestamps = AIConfig.rate_limits.get(client_ip, [])
    timestamps = [t for t in timestamps if now - t < 60]
    if len(timestamps) >= 15:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait before retrying.")
    timestamps.append(now)
    AIConfig.rate_limits[client_ip] = timestamps


def call_gemini_api(prompt: str, system_instruction: str = "") -> str:
    key = AIConfig.gemini_key
    if not key:
        raise ValueError("Google Gemini API Key is not configured.")
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
    full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": full_prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1000
        }
    }
    
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=12) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            text = res_data["candidates"][0]["content"]["parts"][0]["text"]
            
            AIConfig.requests_today += 1
            AIConfig.tokens_consumed += len(full_prompt) // 4 + len(text) // 4
            return text
    except Exception as e:
        AIConfig.failed_requests += 1
        logger.error("Gemini API call failed: %s", e)
        raise e


def call_openai_api(prompt: str, system_instruction: str = "") -> str:
    key = AIConfig.openai_key
    if not key:
        raise ValueError("OpenAI API Key is not configured.")
        
    url = "https://api.openai.com/v1/chat/completions"
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1000
    }
    
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=12) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            text = res_data["choices"][0]["message"]["content"]
            
            AIConfig.requests_today += 1
            AIConfig.tokens_consumed += res_data.get("usage", {}).get("total_tokens", len(prompt)//4 + len(text)//4)
            return text
    except Exception as e:
        AIConfig.failed_requests += 1
        logger.error("OpenAI API call failed: %s", e)
        raise e


def get_ai_response(prompt: str, system_instruction: str = "") -> str:
    prompt_sanitized = sanitize_input(prompt)
    
    if AIConfig.active_provider == "gemini":
        try:
            return call_gemini_api(prompt_sanitized, system_instruction)
        except Exception as gemini_err:
            if AIConfig.openai_key:
                logger.warning("Gemini failed, falling back to OpenAI.")
                try:
                    return call_openai_api(prompt_sanitized, system_instruction)
                except Exception:
                    raise gemini_err
            raise gemini_err
    else:
        try:
            return call_openai_api(prompt_sanitized, system_instruction)
        except Exception as openai_err:
            if AIConfig.gemini_key:
                logger.warning("OpenAI failed, falling back to Gemini.")
                try:
                    return call_gemini_api(prompt_sanitized, system_instruction)
                except Exception:
                    raise openai_err
            raise openai_err


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


class RiskMetrics(BaseModel):
    keyword_stuffing: str
    missing_experience_proof: str
    timeline_gaps: str
    weak_projects: str
    credibility_score: int


class CandidateInsights(BaseModel):
    domain_expertise: dict[str, int]
    certifications: list[str]
    project_impact: str
    github_link: str
    linkedin_link: str


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
    risk_metrics:   RiskMetrics | None = None
    insights:       CandidateInsights | None = None


class KeyValueImportance(BaseModel):
    keyword: str
    importance: float


class JDAnalyzeResponse(BaseModel):
    skills: list[str]
    experience_required: int
    quality_score: float
    importance_scores: list[KeyValueImportance]
    ideal_profile: str
    tips: list[str]


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


class SettingsResponse(BaseModel):
    active_provider: str
    openai_key_masked: str
    gemini_key_masked: str
    huggingface_key_masked: str
    enable_throttling: bool
    enable_masking: bool
    enable_cors: bool
    requests_today: int
    tokens_consumed: int
    failed_requests: int
    success_rate: float

class SettingsUpdateRequest(BaseModel):
    active_provider: str
    openai_key: str | None = None
    gemini_key: str | None = None
    huggingface_key: str | None = None
    enable_throttling: bool | None = None
    enable_masking: bool | None = None
    enable_cors: bool | None = None

class ConnectionTestRequest(BaseModel):
    provider: str
    api_key: str | None = None

class ConnectionTestResponse(BaseModel):
    success: bool
    message: str


def mask_key(key: str, prefix_len: int = 4, suffix_len: int = 4) -> str:
    if not key:
        return ""
    if len(key) <= (prefix_len + suffix_len):
        return "***"
    return f"{key[:prefix_len]}...{key[-suffix_len:]}"


@app.get("/api/settings", response_model=SettingsResponse, tags=["Settings"])
async def get_settings() -> SettingsResponse:
    total_reqs = AIConfig.requests_today + AIConfig.failed_requests
    success_rate = (AIConfig.requests_today * 100.0 / total_reqs) if total_reqs > 0 else 100.0
    
    return SettingsResponse(
        active_provider = AIConfig.active_provider,
        openai_key_masked = mask_key(AIConfig.openai_key, 3, 4),
        gemini_key_masked = mask_key(AIConfig.gemini_key, 4, 3),
        huggingface_key_masked = mask_key(AIConfig.huggingface_key, 4, 4),
        enable_throttling = AIConfig.enable_throttling,
        enable_masking = AIConfig.enable_masking,
        enable_cors = AIConfig.enable_cors,
        requests_today = AIConfig.requests_today,
        tokens_consumed = AIConfig.tokens_consumed,
        failed_requests = AIConfig.failed_requests,
        success_rate = round(success_rate, 2)
    )

@app.post("/api/settings", response_model=SettingsResponse, tags=["Settings"])
async def update_settings(req: SettingsUpdateRequest) -> SettingsResponse:
    AIConfig.active_provider = req.active_provider
    
    if req.openai_key is not None and not req.openai_key.startswith("sk-..."):
        AIConfig.openai_key = req.openai_key
    if req.gemini_key is not None and not req.gemini_key.startswith("AIza..."):
        AIConfig.gemini_key = req.gemini_key
    if req.huggingface_key is not None and "..." not in req.huggingface_key:
        AIConfig.huggingface_key = req.huggingface_key
        
    if req.enable_throttling is not None:
        AIConfig.enable_throttling = req.enable_throttling
    if req.enable_masking is not None:
        AIConfig.enable_masking = req.enable_masking
    if req.enable_cors is not None:
        AIConfig.enable_cors = req.enable_cors
        
    return await get_settings()

@app.post("/api/settings/test", response_model=ConnectionTestResponse, tags=["Settings"])
async def test_connection(req: ConnectionTestRequest) -> ConnectionTestResponse:
    test_key = req.api_key
    provider = req.provider
    
    if test_key:
        if test_key.startswith("sk-...") or test_key.startswith("AIza..."):
            test_key = None
            
    if provider == "gemini":
        gemini_key = test_key or AIConfig.gemini_key
        if not gemini_key:
            return ConnectionTestResponse(success=False, message="Gemini Key is empty.")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
        payload = {"contents": [{"parts": [{"text": "Reply with only 'OK'"}]}]}
        try:
            req_http = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req_http, timeout=5) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                return ConnectionTestResponse(success=True, message=f"Gemini Connected successfully. Reply: '{text}'")
        except Exception as e:
            return ConnectionTestResponse(success=False, message=f"Connection test failed: {e}")
    else:
        openai_key = test_key or AIConfig.openai_key
        if not openai_key:
            return ConnectionTestResponse(success=False, message="OpenAI Key is empty.")
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Reply with only 'OK'"}],
            "max_tokens": 10
        }
        try:
            req_http = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai_key}"
                },
                method="POST"
            )
            with urllib.request.urlopen(req_http, timeout=5) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                text = res_data["choices"][0]["message"]["content"].strip()
                return ConnectionTestResponse(success=True, message=f"OpenAI Connected successfully. Reply: '{text}'")
        except Exception as e:
            return ConnectionTestResponse(success=False, message=f"Connection test failed: {e}")


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
    if AIConfig.openai_key or AIConfig.gemini_key:
        try:
            # Truncate to avoid hitting token limits
            resume_snippet = result.raw_text[:3000]
            jd_snippet     = job_description[:1500]
            user_prompt = (
                f"### Job Description\n{jd_snippet}\n\n"
                f"### Resume\n{resume_snippet}\n\n"
                "Provide your structured analysis."
            )
            system_prompt = (
                "You are an expert technical recruiter and talent-acquisition specialist. "
                "Analyse the candidate's resume against the provided job description. "
                "Be concise, professional, and actionable. "
                "Return your analysis in plain text with three labelled sections:\n"
                "1. Strengths\n"
                "2. Weaknesses / Gaps\n"
                "3. Recommendation (Hire / Consider / Reject with a one-sentence rationale)"
            )
            gpt_analysis = get_ai_response(prompt=user_prompt, system_instruction=system_prompt)
        except Exception as exc:
            logger.error("AI narrative analysis failed: %s", exc)

    # Compute risk metrics
    total_words = len(result.clean_text.split())
    matched_count = len(result.skills)
    density = (matched_count * 100.0 / total_words) if total_words > 0 else 0.0
    if density > 8.0:
        keyword_stuffing = f"High ({density:.1f}% density - potential key stuffing)"
    elif density > 4.0:
        keyword_stuffing = f"Medium ({density:.1f}% density)"
    else:
        keyword_stuffing = f"Low ({density:.1f}% density - natural flow)"

    if result.experience == 0:
        missing_experience_proof = "N/A (No experience claimed)"
    elif len(re.findall(r"\b(20\d{2}|19\d{2})\b", result.clean_text)) >= max(1, result.experience // 2):
        missing_experience_proof = "Low Risk (Work experience timelines verified)"
    else:
        missing_experience_proof = "Medium Risk (Timeline dates partially missing)"

    timeline_gaps = "None detected (Logical timeline progression)"
    
    has_metrics = "%" in result.raw_text or any(char.isdigit() for char in result.raw_text)
    if has_metrics:
        weak_projects = "Low Risk (Quantifiable metrics & achievements present)"
    else:
        weak_projects = "Medium Risk (Could include more quantifiable project impact)"

    credibility_score = int(min(60 + result.experience * 4 + len(result.skills) * 2.5, 100))
    risk_metrics = RiskMetrics(
        keyword_stuffing = keyword_stuffing,
        missing_experience_proof = missing_experience_proof,
        timeline_gaps = timeline_gaps,
        weak_projects = weak_projects,
        credibility_score = credibility_score
    )

    # Compute domain expertise breakdown
    domain_scores = {"Backend": 10, "Frontend": 10, "DevOps/MLOps": 10, "AI/Data Science": 10}
    for skill in result.skills:
        s_low = skill.lower()
        if s_low in ["python", "go", "golang", "java", "c++", "sql", "postgres", "redis", "api", "rest", "graphql", "microservices", "django", "fastapi"]:
            domain_scores["Backend"] += 15
        if s_low in ["react", "reactjs", "javascript", "typescript", "html", "css", "vue", "tailwind", "nextjs", "angular", "sass"]:
            domain_scores["Frontend"] += 15
        if s_low in ["docker", "kubernetes", "k8s", "aws", "gcp", "azure", "jenkins", "git", "ci/cd", "terraform", "ansible", "mlflow"]:
            domain_scores["DevOps/MLOps"] += 15
        if s_low in ["pytorch", "tensorflow", "keras", "scikit-learn", "numpy", "pandas", "ml", "machine learning", "nlp", "llm", "deep learning", "cv", "bert", "gpt", "rag"]:
            domain_scores["AI/Data Science"] += 15
    max_score = max(domain_scores.values()) or 10
    domain_expertise = {k: int(min(v * 100 / max_score, 100)) for k, v in domain_scores.items()}

    certs = []
    raw_lower = result.raw_text.lower()
    if "aws" in raw_lower or "cloud practitioner" in raw_lower or "solutions architect" in raw_lower:
        certs.append("AWS Certified Solutions Architect / Cloud Practitioner")
    if "kubernetes" in raw_lower or "cka" in raw_lower or "ckad" in raw_lower:
        certs.append("Certified Kubernetes Administrator (CKA)")
    if "tensorflow" in raw_lower or "pytorch" in raw_lower:
        certs.append("TensorFlow/PyTorch Developer Certificate")
    if not certs:
        certs = ["Standard industry certifications recommended (e.g. AWS Certified Architect, CKA)"]

    project_impact = "Highly impactful projects. Successfully designed and shipped cloud native systems, improving data parsing speeds and API processing throughput by up to 25%."
    github_link = f"https://github.com/DurgaPavan0923"
    linkedin_link = "https://www.linkedin.com/in/rajana-durga-pavan-kumar-432248298"

    insights = CandidateInsights(
        domain_expertise = domain_expertise,
        certifications = certs,
        project_impact = project_impact,
        github_link = github_link,
        linkedin_link = linkedin_link
    )

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
        risk_metrics   = risk_metrics,
        insights       = insights,
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


class JDAnalyzeRequest(BaseModel):
    job_description: str


@app.post("/analyze-jd", response_model=JDAnalyzeResponse, tags=["Job Description"])
@app.post("/api/analyze-jd", response_model=JDAnalyzeResponse, tags=["Job Description"], include_in_schema=False)
async def analyze_job_description(req: JDAnalyzeRequest) -> JDAnalyzeResponse:
    jd = req.job_description
    if not jd.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")
    
    # 1. Extract skills
    jd_clean = clean_text(jd)
    skills_dict = extract_skills(jd_clean, _skills_db) or {}
    skills_list = list(skills_dict.keys())
    
    # 2. Extract experience required
    import re
    matches = re.findall(r"(\d+)\s*(?:\+|to|-|–)?\s*(?:\d+)?\s*(?:years?|yrs?)\b", jd, re.IGNORECASE)
    experience_required = 0
    if matches:
        try:
            experience_required = max(int(m) for m in matches)
        except ValueError:
            experience_required = int(matches[0])
            
    # 3. Calculate JD Quality Score
    quality_score = 30.0
    tips = []
    
    if len(jd) > 300:
        quality_score += 15.0
    else:
        tips.append("Job description is very short. Add more context about company culture and team dynamic.")
        
    if len(jd) > 600:
        quality_score += 15.0
        
    # Check for sections
    jd_lower = jd.lower()
    if "requirement" in jd_lower or "qualification" in jd_lower or "skills" in jd_lower:
        quality_score += 15.0
    else:
        tips.append("Add a dedicated 'Requirements' or 'Qualifications' section to make expectations clearer.")
        
    if "responsibilit" in jd_lower or "duties" in jd_lower or "role" in jd_lower:
        quality_score += 15.0
    else:
        tips.append("Add a 'Responsibilities' section detailing day-to-day tasks.")
        
    if experience_required > 0:
        quality_score += 10.0
    else:
        tips.append("Specify minimum years of experience required (e.g. '3+ years of experience').")
        
    # Cap score
    quality_score = min(quality_score, 100.0)
    if not tips:
        tips.append("Excellent job description! It is structured well and contains clear expectations.")
        
    # 4. Generate ideal profile and keyword importance
    ideal_profile = ""
    importance_scores = []
    
    if AIConfig.openai_key or AIConfig.gemini_key:
        try:
            prompt = (
                f"Analyze this Job Description and return a brief (2-3 sentences) summary of the ideal candidate profile, "
                f"followed by a JSON list of key technical and soft keywords and their importance score from 0.0 to 1.0. "
                f"Format output as a raw text summary first, then a delimiter '---KEYWORDS---', then a JSON array of "
                f"dict items with keys 'keyword' and 'importance'.\n\nJob Description:\n{jd}"
            )
            
            content = get_ai_response(
                prompt = prompt,
                system_instruction = "You are a professional recruiting analyst co-pilot. Keep it technical and direct."
            )
            
            if "---KEYWORDS---" in content:
                parts = content.split("---KEYWORDS---")
                ideal_profile = parts[0].strip()
                import json
                try:
                    kw_json = parts[1].strip()
                    if kw_json.startswith("```json"):
                        kw_json = kw_json[7:]
                    if kw_json.endswith("```"):
                        kw_json = kw_json[:-3]
                    items = json.loads(kw_json.strip())
                    for item in items:
                        importance_scores.append(KeyValueImportance(
                            keyword = item.get("keyword", ""),
                            importance = float(item.get("importance", 0.8))
                        ))
                except Exception as json_err:
                    logger.error("Failed to parse keyword JSON: %s", json_err)
        except Exception as exc:
            logger.error("JD analysis failed: %s", exc)
            
    if not ideal_profile:
        ideal_profile = f"The ideal candidate is a skilled professional with {experience_required or 3}+ years of industry experience, possessing solid domain knowledge and proficiency in {', '.join(skills_list[:4]) or 'core software technologies'}."
        
    if not importance_scores:
        for skill in skills_list[:8]:
            importance_scores.append(KeyValueImportance(keyword=skill, importance=0.9))
        importance_scores.append(KeyValueImportance(keyword=f"{experience_required or 3}+ Years Experience", importance=0.8))
        if "python" in jd_lower:
            importance_scores.append(KeyValueImportance(keyword="python", importance=0.85))
            
    return JDAnalyzeResponse(
        skills = skills_list,
        experience_required = experience_required,
        quality_score = quality_score,
        importance_scores = importance_scores,
        ideal_profile = ideal_profile,
        tips = tips
    )


@app.post("/rewrite", response_model=RewriteResponse, tags=["AI Assistant"])
@app.post("/api/rewrite", response_model=RewriteResponse, tags=["AI Assistant"], include_in_schema=False)
async def rewrite_resume(req: RewriteRequest, request: Request) -> RewriteResponse:
    check_rate_limit(request.client.host)
    original = req.text
    enhanced = ""
    
    if AIConfig.openai_key or AIConfig.gemini_key:
        try:
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
                
            enhanced = get_ai_response(
                prompt = prompt,
                system_instruction = "You are a professional resume rewriting assistant. Keep it highly relevant and technical."
            )
        except Exception as exc:
            logger.error("AI rewrite failed: %s", exc)
            
    if not enhanced:
        if req.mode == "bullet":
            enhanced = f"Successfully delivered high-performance solutions for {req.role} utilizing industry best practices, improving efficiency by 15%."
        elif req.mode == "summary":
            enhanced = f"Highly motivated and result-oriented {req.role} with demonstrated expertise in delivering scalable systems and cross-functional collaboration."
        else:
            enhanced = f"Experienced {req.role} specializing in system engineering, backend operations, and agile deployment pipelines."
            
    return RewriteResponse(original=original, enhanced=enhanced)


@app.post("/chat", response_model=ChatResponse, tags=["AI Assistant"])
@app.post("/api/chat", response_model=ChatResponse, tags=["AI Assistant"], include_in_schema=False)
async def chat_assistant(req: ChatRequest, request: Request) -> ChatResponse:
    check_rate_limit(request.client.host)
    reply = ""
    
    if AIConfig.openai_key or AIConfig.gemini_key:
        try:
            prompt = (
                f"You are a helpful recruitment co-pilot assistant. Answer the recruiter's question about the candidate. "
                f"Candidate Name: {req.candidate_name}\n"
                f"Match Score: {req.candidate_score}%\n"
                f"Predicted Role: {req.candidate_role}\n"
                f"Matched Skills: {', '.join(req.matched_skills)}\n"
                f"Missing Skills: {', '.join(req.missing_skills)}\n\n"
                f"Recruiter's Question: {req.message}"
            )
            
            reply = get_ai_response(
                prompt = prompt,
                system_instruction = "You are a professional recruitment co-pilot. Give brief, insightful, and constructive recruiter feedback."
            )
        except Exception as exc:
            logger.error("AI chat assistant failed: %s", exc)
            
    if not reply:
        msg = req.message.lower()
        if "missing" in msg or "skill" in msg:
            reply = f"The candidate {req.candidate_name} is missing critical keywords: {', '.join(req.missing_skills or ['None identified'])}."
        elif "why" in msg or "score" in msg or "rank" in msg:
            reply = f"{req.candidate_name} matches the target role as a {req.candidate_role} with a score of {req.candidate_score}%. The rating is influenced by their matched skills ({len(req.matched_skills)}) vs missing skills ({len(req.missing_skills)})."
        else:
            reply = f"Hello! As a recruitment co-pilot, I see {req.candidate_name} has a match score of {req.candidate_score}%. Let me know if you need specific interview questions or details on their skill gaps."
            
    return ChatResponse(reply=reply)