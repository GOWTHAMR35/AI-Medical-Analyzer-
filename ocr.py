"""
ocr.py - Text extraction from PDFs and images.
Handles both PDF (via pdfplumber/PyMuPDF) and image (via pytesseract) inputs.
"""

import io
import os
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract plain text from a PDF file.
    Tries pdfplumber first (better for text-layer PDFs), falls back to PyMuPDF.
    """
    text = ""

    # --- Attempt 1: pdfplumber (great for text-based PDFs) ---
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages_text = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
            text = "\n".join(pages_text)
    except Exception as e:
        print(f"[pdfplumber] extraction failed: {e}")

    # --- Attempt 2: PyMuPDF (handles more PDF variants) ---
    if not text.strip():
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            pages_text = []
            for page in doc:
                pages_text.append(page.get_text())
            text = "\n".join(pages_text)
            doc.close()
        except Exception as e:
            print(f"[PyMuPDF] extraction failed: {e}")

    return text.strip()


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------

def extract_text_from_image(file_bytes: bytes) -> str:
    """
    Extract text from an image using pytesseract (OCR).
    Falls back gracefully if tesseract is not installed.
    """
    try:
        import pytesseract
        from PIL import Image

        image = Image.open(io.BytesIO(file_bytes))

        # Convert to RGB if needed (handles RGBA / palette modes)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        # Run OCR with English language pack
        text = pytesseract.image_to_string(image, lang="eng")
        return text.strip()

    except pytesseract.TesseractNotFoundError:
        return (
            "[OCR ERROR] Tesseract is not installed on this system.\n"
            "Please install it: https://github.com/tesseract-ocr/tesseract\n"
            "On Ubuntu: sudo apt-get install tesseract-ocr\n"
            "On macOS: brew install tesseract"
        )
    except Exception as e:
        return f"[OCR ERROR] {str(e)}"


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Dispatch to the correct extractor based on file extension.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename (used to detect type).

    Returns:
        Extracted text string.
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"):
        return extract_text_from_image(file_bytes)
    else:
        return f"[ERROR] Unsupported file type: '{ext}'. Please upload a PDF or image."
