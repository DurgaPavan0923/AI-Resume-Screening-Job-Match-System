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


def _extract_docx(raw_bytes: bytes) -> str:
    import zipfile
    import xml.etree.ElementTree as ET
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
            if "word/document.xml" in z.namelist():
                xml_content = z.read("word/document.xml")
                tree = ET.fromstring(xml_content)
                texts = [node.text for node in tree.iter() if node.tag.endswith("}t") and node.text]
                return "\n".join(texts)
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_pdf(file) -> str:
    """
    Extract text from a PDF or DOCX file.

    Parameters
    ----------
    file : file-like object or Streamlit UploadedFile
        The PDF/DOCX source.  ``seek(0)`` is called before reading.

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

    # Check for DOCX container (ZIP starting with PK)
    if raw.startswith(b"PK\x03\x04"):
        docx_text = _extract_docx(raw)
        if docx_text.strip():
            return docx_text

    for extractor in (_extract_pdfplumber, _extract_pypdf2):
        try:
            result = extractor(io.BytesIO(raw))
            if result.strip():
                return result
        except Exception as exc:
            logger.debug("Extractor %s failed: %s", extractor.__name__, exc)

    logger.warning("All PDF/DOCX extractors failed — returning empty string.")
    return ""