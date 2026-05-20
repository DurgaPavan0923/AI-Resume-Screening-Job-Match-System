"""
src/experience_extractor.py — Heuristic extraction of total years of
professional experience from resume text.
"""

from __future__ import annotations

import re

# Patterns ordered from most to least specific
_PATTERNS: list[re.Pattern] = [
    # "10+ years", "5-7 years of experience"
    re.compile(
        r"(\d+)\s*[\+\-–]\s*(?:\d+\s*)?years?\s*(?:of\s*)?(?:experience|exp)",
        re.IGNORECASE,
    ),
    # "over 8 years"
    re.compile(r"over\s+(\d+)\s+years?", re.IGNORECASE),
    # "3 years experience"
    re.compile(
        r"(\d+)\s+years?\s*(?:of\s*)?(?:experience|exp|work)",
        re.IGNORECASE,
    ),
    # "experience of 6 years"
    re.compile(r"experience\s+of\s+(\d+)\s+years?", re.IGNORECASE),
    # Standalone digit followed by "yrs"
    re.compile(r"(\d+)\s*yrs?", re.IGNORECASE),
]

_MAX_PLAUSIBLE = 50   # discard obviously wrong numbers


def extract_experience(text: str) -> int:
    """
    Return the most plausible years-of-experience figure found in *text*.

    Returns 0 when nothing is detected.
    """
    if not isinstance(text, str):
        return 0

    candidates: list[int] = []
    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            try:
                value = int(match.group(1))
                if 0 < value <= _MAX_PLAUSIBLE:
                    candidates.append(value)
            except (IndexError, ValueError):
                continue

    return max(candidates) if candidates else 0