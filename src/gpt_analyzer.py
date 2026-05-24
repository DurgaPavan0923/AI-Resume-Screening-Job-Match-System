"""
src/gpt_analyzer.py — Optional GPT/OpenAI-powered resume analysis.

Set the environment variable ``OPENAI_API_KEY`` (or add it to
Streamlit secrets as ``openai_api_key``) to enable this feature.
The rest of the app degrades gracefully when the key is absent.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert technical recruiter and talent-acquisition specialist. "
    "Analyze the candidate's resume against the provided job description. "
    "Be concise, professional, and recruiter-focused. Do NOT use boilerplate introductions or pleasantries. "
    "Return your analysis in plain text with exactly these seven labeled sections:\n"
    "1. Executive Summary: Concisely summarize candidate fit, background alignment, and overall suitability.\n"
    "2. Technical Strengths: Bullet points identifying key technical match items and domains.\n"
    "3. Missing Capabilities: Bullet points listing technical requirements not found in the resume.\n"
    "4. Career Progression: Professional trajectory progression, stability, and growth path.\n"
    "5. Risk Indicators: Red flags, timeline inconsistencies, keyword stuffing density, or training gaps.\n"
    "6. Interview Focus Areas: 3 custom topics/questions to verify missing or weak capabilities.\n"
    "7. Final Hiring Recommendation: Actionable hiring decision ([Strong Hire], [Hire], [Consider], or [Reject]) and a one-sentence justification."
)


def _get_api_key() -> str | None:
    """Retrieve the OpenAI API key from env or Streamlit secrets."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st  # noqa: PLC0415

        return st.secrets.get("openai_api_key")
    except Exception:
        return None


def analyze_resume(resume_text: str, job_description: str) -> str:
    """
    Call the OpenAI Chat Completions API and return an analysis string.

    Raises
    ------
    RuntimeError
        If no API key is configured or the API call fails, so that
        ``app.py`` can catch the exception and show a fallback message.
    """
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")

    try:
        from openai import OpenAI  # type: ignore  # noqa: PLC0415

        client = OpenAI(api_key=api_key)

        # Truncate to avoid hitting token limits
        resume_snippet = resume_text[:3_000]
        jd_snippet     = job_description[:1_500]

        user_prompt = (
            f"### Job Description\n{jd_snippet}\n\n"
            f"### Resume\n{resume_snippet}\n\n"
            "Provide your structured analysis."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=600,
            temperature=0.4,
        )

        return response.choices[0].message.content.strip()

    except Exception as exc:
        logger.error("GPT analysis failed: %s", exc)
        raise RuntimeError(str(exc)) from exc