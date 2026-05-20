"""
src/pdf_parser.py — Extract plain text from uploaded PDF files.

Tries pdfplumber first (best layout fidelity), falls back to
PyPDF2, then pdfminer.six if available.
"""

from __future__ import annotations

import io
import logging
from typing import BinaryIO

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary extractor — pdfplumber
# ---------------------------------------------------------------------------
def _extract_pdfplumber(file_like: BinaryIO) -> str:
    import pdfplumber  # type: ignore

    pages: list[str] = []
    with pdfplumber.open(file_like) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Fallback — PyPDF2
# ---------------------------------------------------------------------------
def _extract_pypdf2(file_like: BinaryIO) -> str:
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except ImportError:
        from PyPDF2 import PdfFileReader as PdfReader  # type: ignore (older API)

    reader = PdfReader(file_like)
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_pdf(file) -> str:
    """
    Extract text from a PDF.

    Parameters
    ----------
    file : file-like object or Streamlit UploadedFile
        The PDF source.  ``seek(0)`` is called before reading.

    Returns
    -------
    str
        Extracted plain text, or an empty string if extraction fails.
    """
    # Streamlit UploadedFile → read bytes, then wrap in BytesIO
    if hasattr(file, "read"):
        file.seek(0)
        raw = file.read()
    else:
        with open(file, "rb") as fh:
            raw = fh.read()

    for extractor in (_extract_pdfplumber, _extract_pypdf2):
        try:
            result = extractor(io.BytesIO(raw))
            if result.strip():
                return result
        except Exception as exc:
            logger.debug("Extractor %s failed: %s", extractor.__name__, exc)

    logger.warning("All PDF extractors failed — returning empty string.")
    return ""