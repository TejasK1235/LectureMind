# question_bank/slide_ingestor.py

import os
from typing import List, Dict, Optional


def ingest_slides(file_path: str) -> List[Dict]:
    """
    Extracts text from each slide/page of a PPT, PPTX, or PDF file.

    Returns a list of dicts:
    [
        {"slide_number": 1, "text": "extracted text from slide 1"},
        {"slide_number": 2, "text": "extracted text from slide 2"},
        ...
    ]

    Slides with no meaningful text are excluded.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Slide file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".ppt", ".pptx"):
        return _ingest_pptx(file_path)
    elif ext == ".pdf":
        return _ingest_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use PDF or PPTX.")


def _ingest_pptx(file_path: str) -> List[Dict]:
    """Extract text from each slide in a PPTX file."""
    try:
        from pptx import Presentation
    except ImportError:
        raise ImportError("python-pptx not installed. Run: pip install python-pptx")

    prs = Presentation(file_path)
    slides = []

    for i, slide in enumerate(prs.slides, start=1):
        texts = []

        for shape in slide.shapes:
            # Only process shapes that have a text frame
            if not shape.has_text_frame:
                continue

            for para in shape.text_frame.paragraphs:
                line = " ".join(
                    run.text.strip()
                    for run in para.runs
                    if run.text.strip()
                )
                if line:
                    texts.append(line)

        combined = " ".join(texts).strip()

        # Skip slides with no meaningful text
        if len(combined.split()) < 4:
            continue

        slides.append({
            "slide_number": i,
            "text": combined
        })

    return slides


def _ingest_pdf(file_path: str) -> List[Dict]:
    """Extract text from each page in a PDF file."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    doc = fitz.open(file_path)
    slides = []

    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()

        # Clean up excessive whitespace and newlines
        text = " ".join(text.split())

        # Skip pages with no meaningful text
        if len(text.split()) < 4:
            continue

        slides.append({
            "slide_number": i,
            "text": text
        })

    doc.close()
    return slides