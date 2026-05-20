"""
src/explainer_llm.py — LLM-based natural-language explainer.

Uses the OpenAI API (if configured) to produce a recruiter-friendly
plain-English explanation of the match score and skill gaps.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful recruiting assistant. "
    "Given a candidate's match score, matched skills, and skill gaps, "
    "write 2–3 concise sentences explaining the result to a recruiter. "
    "Be factual, professional, and encouraging where appropriate."
)


def explain_match(
    score: float,
    matched_skills: list[str],
    missing_skills: list[str],
    role: str,
) -> str:
    """
    Generate a natural-language explanation of a candidate's match result.

    Falls back to a template string when the API is unavailable.

    Parameters
    ----------
    score : float
        Overall match percentage (0–100).
    matched_skills : list[str]
        Skills the candidate has that appear in the JD.
    missing_skills : list[str]
        Skills required by the JD that are absent from the resume.
    role : str
        Predicted job role label.

    Returns
    -------
    str
        Human-readable explanation.
    """
    # --- Fallback template (no API key required) ---
    def _template_explanation() -> str:
        ms  = ", ".join(matched_skills[:5]) or "none detected"
        gap = ", ".join(missing_skills[:5]) or "none"
        verdict = (
            "a strong match" if score >= 75
            else "a partial match" if score >= 50
            else "a below-threshold match"
        )
        return (
            f"This candidate is {verdict} for the {role} role "
            f"with an overall score of {score:.1f}%. "
            f"Key matched skills include: {ms}. "
            f"Notable gaps: {gap}."
        )

    try:
        from openai import OpenAI  # type: ignore  # noqa: PLC0415
        import os  # noqa: PLC0415

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return _template_explanation()

        client = OpenAI(api_key=api_key)

        user_prompt = (
            f"Role: {role}\n"
            f"Match score: {score:.1f}%\n"
            f"Matched skills: {', '.join(matched_skills[:8]) or 'none'}\n"
            f"Missing skills: {', '.join(missing_skills[:8]) or 'none'}\n\n"
            "Write a concise recruiter explanation."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=180,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()

    except Exception as exc:
        logger.debug("LLM explainer unavailable (%s) — using template.", exc)
        return _template_explanation()