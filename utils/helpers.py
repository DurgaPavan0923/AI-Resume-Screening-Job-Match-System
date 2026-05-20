"""
utils/helpers.py — Shared utility functions for validation and formatting.
"""

from __future__ import annotations


def validate_input(job_desc: str, files) -> tuple[bool, str]:
    """
    Validate user inputs before processing.

    Returns
    -------
    tuple[bool, str]
        ``(is_valid, message)`` — message is empty when valid.
    """
    if not job_desc or not job_desc.strip():
        return False, "⚠️ Please paste a job description before analysing."

    if len(job_desc.strip()) < 50:
        return False, "⚠️ Job description is too short. Please provide more detail."

    if not files:
        return False, "⚠️ Please upload at least one PDF resume."

    return True, ""


def format_skills(skills: dict | list | None, max_display: int = 12) -> str:
    """
    Convert a skills dict or list to a comma-separated display string.

    Parameters
    ----------
    skills : dict or list or None
    max_display : int
        Cap the number of skills shown to avoid UI overflow.

    Returns
    -------
    str
        Formatted skill string, e.g. ``"Python, SQL, TensorFlow, +3 more"``
    """
    if not skills:
        return "None detected"

    skill_names: list[str] = (
        list(skills.keys()) if isinstance(skills, dict) else list(skills)
    )

    # Title-case for display
    display = [s.title() for s in skill_names[:max_display]]
    result  = ", ".join(display)

    overflow = len(skill_names) - max_display
    if overflow > 0:
        result += f", +{overflow} more"

    return result


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float between *lo* and *hi*."""
    return max(lo, min(hi, value))


def percentage_bar(value: float, width: int = 20) -> str:
    """
    Return a simple ASCII progress bar string.

    Example: ``[████████░░░░░░░░░░░░]  40%``
    """
    filled = int(round(value / 100 * width))
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}]  {value:.1f}%"