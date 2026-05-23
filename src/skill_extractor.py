"""
src/skill_extractor.py — Load a skill vocabulary and extract matched skills
from cleaned resume / job-description text.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def load_skills(path: str | Path) -> dict[str, int]:
    """
    Load skills from a plain-text file (one skill per line).

    Returns
    -------
    dict[str, int]
        Mapping of lower-cased skill name → weight (default 1).
        Lines starting with ``#`` are treated as comments.
        Optionally supports ``skill:weight`` syntax.
    """
    skills: dict[str, int] = {}
    path = Path(path)

    if not path.exists():
        logger.warning("Skills file not found: %s", path)
        return skills

    with path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                parts = line.split(":", 1)
                name   = parts[0].strip().lower()
                try:
                    weight = int(parts[1].strip())
                except ValueError:
                    weight = 1
            else:
                name   = line.lower()
                weight = 1
            if name:
                skills[name] = weight

    logger.info("Loaded %d skills from %s", len(skills), path)
    return skills


SYNONYMS = {
    r"\breactjs\b": "react",
    r"\breact\.js\b": "react",
    r"\bml\b": "machine learning",
    r"\bdl\b": "deep learning",
    r"\bk8s\b": "kubernetes",
    r"\baws\b": "amazon web services",
    r"\bgcp\b": "google cloud platform",
    r"\bazure\b": "microsoft azure",
    r"\bjs\b": "javascript",
    r"\bts\b": "typescript",
    r"\bnodejs\b": "node.js",
    r"\bpython3\b": "python",
    r"\bnlp\b": "natural language processing",
    r"\bcv\b": "computer vision",
    r"\bai\b": "artificial intelligence"
}

def expand_synonyms(text: str) -> str:
    text_lower = text.lower()
    for pattern, canonical in SYNONYMS.items():
        text_lower = re.sub(pattern, canonical, text_lower)
    return text_lower


def extract_skills(text: str, skills_db: dict[str, int]) -> dict[str, int]:
    """
    Find which skills from ``skills_db`` are mentioned in ``text``.

    Uses whole-word matching so "r" does not match inside "recruit".

    Returns
    -------
    dict[str, int]
        Subset of ``skills_db`` containing only matched skills → weight.
    """
    if not text or not skills_db:
        return {}

    text_lower = expand_synonyms(text.lower())
    matched: dict[str, int] = {}

    for skill, weight in skills_db.items():
        # Escape hyphens etc. for regex; match whole word/phrase
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            matched[skill] = weight

    return matched