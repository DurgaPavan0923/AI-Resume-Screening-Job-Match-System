"""
config.py — Central configuration for AI Resume Screening System.
All path constants and tunable parameters live here.
"""

import os

# ── Base directory (project root) ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data paths ─────────────────────────────────────────────────
DATA_DIR        = os.path.join(BASE_DIR, "data")
SKILLS_PATH     = os.path.join(DATA_DIR, "skills.txt")
JOB_ROLES_PATH  = os.path.join(DATA_DIR, "job_roles.csv")

# ── Asset paths ────────────────────────────────────────────────
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
CSS_PATH   = os.path.join(ASSETS_DIR, "styles.css")

# ── Scoring weights (must sum to 1.0) ──────────────────────────
WEIGHT_SIMILARITY  = 0.50
WEIGHT_SKILL       = 0.30
WEIGHT_EXPERIENCE  = 0.20

# ── Hiring thresholds ──────────────────────────────────────────
THRESHOLD_HIRE     = 75   # score >= this → Hire
THRESHOLD_CONSIDER = 50   # score >= this → Consider (else Reject)

# ── Experience cap for normalisation ───────────────────────────
MAX_EXPERIENCE_YEARS = 10

# ── Model / vectoriser settings ────────────────────────────────
VECTORIZER_MAX_FEATURES = 5_000
RANDOM_STATE            = 42