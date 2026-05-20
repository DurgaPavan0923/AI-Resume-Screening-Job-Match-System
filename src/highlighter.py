"""
src/highlighter.py — Highlight matched keywords in resume text
for display in the Streamlit UI.
"""

from __future__ import annotations

import html
import re


def highlight_text(
    text: str,
    keywords: list[str],
    color: str = "#00e5ff",
    bg_color: str = "rgba(0,229,255,0.15)",
) -> str:
    """
    Wrap every occurrence of each keyword in *text* with an HTML
    ``<mark>`` span using the provided colours.

    Parameters
    ----------
    text : str
        Plain text to process (will be HTML-escaped first).
    keywords : list[str]
        Words / phrases to highlight (case-insensitive).
    color : str
        CSS colour for the highlighted text.
    bg_color : str
        CSS background colour for the highlight.

    Returns
    -------
    str
        HTML string safe to pass to ``st.markdown(..., unsafe_allow_html=True)``.
    """
    if not text or not keywords:
        return html.escape(text)

    escaped = html.escape(text)

    mark_style = (
        f"color:{color};"
        f"background:{bg_color};"
        "font-weight:600;"
        "border-radius:3px;"
        "padding:0 2px;"
    )

    # Sort longest keyword first to avoid partial replacements
    for kw in sorted(keywords, key=len, reverse=True):
        pattern = re.compile(re.escape(html.escape(kw)), re.IGNORECASE)
        escaped = pattern.sub(
            lambda m: f'<mark style="{mark_style}">{m.group(0)}</mark>',
            escaped,
        )

    return escaped