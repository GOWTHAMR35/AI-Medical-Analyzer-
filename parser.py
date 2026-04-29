"""
parser.py - Extract structured medical parameters from raw OCR text.

Uses regex patterns to identify numeric values associated with common
lab test names. Designed to be robust against varied report formats.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MedicalParameter:
    name: str                    # Human-readable name
    value: Optional[float]       # Numeric value (None if not found)
    unit: str                    # Unit string (e.g., "mg/dL")
    raw_match: str = ""          # The raw text snippet that matched


# ---------------------------------------------------------------------------
# Parameter definitions
# Each entry: (display_name, unit, [regex_pattern, ...])
# Patterns must capture the numeric value in group 1.
# ---------------------------------------------------------------------------

PARAMETER_PATTERNS = {
    "glucose": (
        "Blood Glucose",
        "mg/dL",
        [
            r"(?:fasting\s+)?(?:blood\s+)?glucose[\s:=]+(\d+\.?\d*)",
            r"(?:fbs|rbs|ppbs)[\s:=]+(\d+\.?\d*)",
            r"glucose[\s\S]{0,30}?(\d{2,3})\s*mg",
        ],
    ),
    "hemoglobin": (
        "Hemoglobin",
        "g/dL",
        [
            r"h(?:ae|e)moglobin[\s:=]+(\d+\.?\d*)",
            r"\bhb\b[\s:=]+(\d+\.?\d*)",
            r"hgb[\s:=]+(\d+\.?\d*)",
        ],
    ),
    "cholesterol_total": (
        "Total Cholesterol",
        "mg/dL",
        [
            r"total\s+cholesterol[\s:=]+(\d+\.?\d*)",
            r"cholesterol[\s,]+total[\s:=]+(\d+\.?\d*)",
            r"(?<!hdl\s)(?<!ldl\s)cholesterol[\s:=]+(\d+\.?\d*)",
        ],
    ),
    "hdl": (
        "HDL Cholesterol",
        "mg/dL",
        [
            r"hdl[\s\-]?(?:cholesterol)?[\s:=]+(\d+\.?\d*)",
            r"high[\s\-]density[\s\S]{0,20}?(\d+\.?\d*)",
        ],
    ),
    "ldl": (
        "LDL Cholesterol",
        "mg/dL",
        [
            r"ldl[\s\-]?(?:cholesterol)?[\s:=]+(\d+\.?\d*)",
            r"low[\s\-]density[\s\S]{0,20}?(\d+\.?\d*)",
        ],
    ),
    "triglycerides": (
        "Triglycerides",
        "mg/dL",
        [
            r"triglycerides?[\s:=]+(\d+\.?\d*)",
            r"tg[\s:=]+(\d+\.?\d*)",
        ],
    ),
    "systolic_bp": (
        "Systolic BP",
        "mmHg",
        [
            r"(?:blood\s+pressure|bp)[\s:=]+(\d{2,3})\s*/\s*\d{2,3}",
            r"systolic[\s:=]+(\d{2,3})",
            r"(\d{2,3})\s*/\s*\d{2,3}\s*mmhg",
        ],
    ),
    "diastolic_bp": (
        "Diastolic BP",
        "mmHg",
        [
            r"(?:blood\s+pressure|bp)[\s:=]+\d{2,3}\s*/\s*(\d{2,3})",
            r"diastolic[\s:=]+(\d{2,3})",
            r"\d{2,3}\s*/\s*(\d{2,3})\s*mmhg",
        ],
    ),
    "creatinine": (
        "Creatinine",
        "mg/dL",
        [
            r"creatinine[\s:=]+(\d+\.?\d*)",
            r"serum\s+creatinine[\s:=]+(\d+\.?\d*)",
        ],
    ),
    "urea": (
        "Blood Urea",
        "mg/dL",
        [
            r"(?:blood\s+)?urea[\s:=]+(\d+\.?\d*)",
            r"bun[\s:=]+(\d+\.?\d*)",
        ],
    ),
    "wbc": (
        "WBC Count",
        "cells/μL",
        [
            r"(?:wbc|white\s+blood\s+(?:cell|count))[\s:=]+(\d+\.?\d*)",
            r"leukocytes?[\s:=]+(\d+\.?\d*)",
        ],
    ),
    "rbc": (
        "RBC Count",
        "million/μL",
        [
            r"(?:rbc|red\s+blood\s+(?:cell|count))[\s:=]+(\d+\.?\d*)",
            r"erythrocytes?[\s:=]+(\d+\.?\d*)",
        ],
    ),
    "platelets": (
        "Platelet Count",
        "thousand/μL",
        [
            r"platelets?(?:\s+count)?[\s:=]+(\d+\.?\d*)",
            r"thrombocytes?[\s:=]+(\d+\.?\d*)",
        ],
    ),
    "tsh": (
        "TSH",
        "mIU/L",
        [
            r"tsh[\s:=]+(\d+\.?\d*)",
            r"thyroid[\s\S]{0,30}?stimulating[\s\S]{0,20}?(\d+\.?\d*)",
        ],
    ),
    "hba1c": (
        "HbA1c",
        "%",
        [
            r"hba1c[\s:=]+(\d+\.?\d*)",
            r"glycated\s+h(?:ae|e)moglobin[\s:=]+(\d+\.?\d*)",
            r"a1c[\s:=]+(\d+\.?\d*)",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Parsing logic
# ---------------------------------------------------------------------------

def extract_parameters(text: str) -> dict[str, MedicalParameter]:
    """
    Scan extracted OCR text for known medical parameters.

    Args:
        text: Raw text from OCR.

    Returns:
        Dict mapping parameter key → MedicalParameter.
        Only includes parameters that were actually found.
    """
    text_lower = text.lower()
    found: dict[str, MedicalParameter] = {}

    for key, (display_name, unit, patterns) in PARAMETER_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    value = float(match.group(1))
                    found[key] = MedicalParameter(
                        name=display_name,
                        value=value,
                        unit=unit,
                        raw_match=match.group(0),
                    )
                    break  # Use the first matching pattern
                except (ValueError, IndexError):
                    continue

    return found


def summarize_extracted(params: dict[str, MedicalParameter]) -> str:
    """
    Create a concise text summary of extracted parameters
    (used as context for the LLM prompt).
    """
    if not params:
        return "No standard medical parameters could be extracted."

    lines = []
    for key, param in params.items():
        lines.append(f"- {param.name}: {param.value} {param.unit}")
    return "\n".join(lines)
