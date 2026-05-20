"""
src/education_parser.py — Extract degree / qualification mentions
from resume text using keyword matching.
"""

from __future__ import annotations

import re

# Map of regex pattern → canonical label
_DEGREE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bph\.?d\b", re.IGNORECASE),                        "PhD"),
    (re.compile(r"\bdoctor(?:ate)?\b", re.IGNORECASE),                "Doctorate"),
    (re.compile(r"\bm\.?tech\b", re.IGNORECASE),                      "M.Tech"),
    (re.compile(r"\bm\.?e\.?\b", re.IGNORECASE),                      "M.E."),
    (re.compile(r"\bm\.?sc?\b", re.IGNORECASE),                       "M.Sc"),
    (re.compile(r"\bmaster(?:'?s)?\b", re.IGNORECASE),                "Master's"),
    (re.compile(r"\bmba\b", re.IGNORECASE),                           "MBA"),
    (re.compile(r"\bb\.?tech\b", re.IGNORECASE),                      "B.Tech"),
    (re.compile(r"\bb\.?e\.?\b", re.IGNORECASE),                      "B.E."),
    (re.compile(r"\bb\.?sc?\b", re.IGNORECASE),                       "B.Sc"),
    (re.compile(r"\bbachelor(?:'?s)?\b", re.IGNORECASE),              "Bachelor's"),
    (re.compile(r"\bb\.?c\.?a\b", re.IGNORECASE),                     "BCA"),
    (re.compile(r"\bm\.?c\.?a\b", re.IGNORECASE),                     "MCA"),
    (re.compile(r"\bassociate(?:\s+degree)?\b", re.IGNORECASE),       "Associate Degree"),
    (re.compile(r"\bdiploma\b", re.IGNORECASE),                       "Diploma"),
    (re.compile(r"\bhigh\s+school\b", re.IGNORECASE),                 "High School"),
    (re.compile(r"\b12(?:th)?\s+(?:pass|grade|standard)\b", re.IGNORECASE), "12th Grade"),
    (re.compile(r"\b10(?:th)?\s+(?:pass|grade|standard)\b", re.IGNORECASE), "10th Grade"),
]


def extract_education(text: str) -> list[str]:
    """
    Return a deduplicated list of qualifications found in *text*,
    ordered from highest to lowest (as defined in ``_DEGREE_PATTERNS``).
    """
    if not isinstance(text, str):
        return []

    found: list[str] = []
    for pattern, label in _DEGREE_PATTERNS:
        if pattern.search(text) and label not in found:
            found.append(label)

    return found