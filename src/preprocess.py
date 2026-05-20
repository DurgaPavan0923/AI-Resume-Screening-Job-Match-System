"""
src/preprocess.py — Text cleaning and normalisation utilities.
"""

import re
import string

# ---------------------------------------------------------------------------
# Optional: use NLTK stopwords when available, fall back to a minimal set
# ---------------------------------------------------------------------------
try:
    from nltk.corpus import stopwords as _nltk_sw
    _STOPWORDS: set[str] = set(_nltk_sw.words("english"))
except Exception:
    _STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "was", "are", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall",
        "not", "no", "nor", "so", "yet", "both", "either", "neither",
        "each", "few", "more", "most", "other", "some", "such", "than",
        "too", "very", "just", "as", "if", "this", "that", "these",
        "those", "it", "its", "i", "me", "my", "we", "our", "you",
        "your", "he", "she", "they", "them", "their",
    }

_WHITESPACE_RE  = re.compile(r"\s+")
_NON_ALPHA_RE   = re.compile(r"[^a-z0-9\s\-]")
_URL_RE         = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE       = re.compile(r"\S+@\S+\.\S+")
_PHONE_RE       = re.compile(r"\+?\d[\d\s\-().]{7,}\d")


def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Normalise raw text extracted from a resume or job description.

    Steps
    -----
    1. Lower-case
    2. Strip URLs, e-mails, phone numbers
    3. Remove punctuation (keep hyphens — useful for skill names)
    4. Collapse whitespace
    5. Optionally remove stopwords
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _PHONE_RE.sub(" ", text)
    # Remove everything except letters, digits, spaces, hyphens
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    if remove_stopwords:
        text = " ".join(w for w in text.split() if w not in _STOPWORDS)

    return text


def tokenise(text: str) -> list[str]:
    """Return a list of lower-cased tokens, stopwords excluded."""
    cleaned = clean_text(text, remove_stopwords=True)
    return cleaned.split()