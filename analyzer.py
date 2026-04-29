"""
analyzer.py - Rule-based classification of medical parameter values.

Each parameter has evidence-based reference ranges.
Returns Low / Normal / High classification with severity coloring.
"""

from dataclasses import dataclass
from typing import Optional
from utils.parser import MedicalParameter

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AnalyzedParameter:
    name: str
    value: float
    unit: str
    status: str          # "Low" | "Normal" | "High" | "Unknown"
    normal_range: str    # Human-readable range string
    color: str           # CSS/Streamlit color hint
    icon: str            # Emoji indicator
    severity: int        # 0=normal, 1=mild, 2=significant (for sorting)


# ---------------------------------------------------------------------------
# Reference ranges
# Format: key → (low_threshold, high_threshold, normal_range_text)
# None means "no lower / upper bound"
# ---------------------------------------------------------------------------

REFERENCE_RANGES = {
    # Glucose (fasting)
    "glucose": (70.0, 99.0, "70–99 mg/dL (fasting)"),

    # Hemoglobin (adult, combined threshold)
    "hemoglobin": (12.0, 17.5, "12–17.5 g/dL"),

    # Cholesterol
    "cholesterol_total": (None, 200.0, "< 200 mg/dL"),
    "hdl": (40.0, None, "> 40 mg/dL"),
    "ldl": (None, 100.0, "< 100 mg/dL"),
    "triglycerides": (None, 150.0, "< 150 mg/dL"),

    # Blood pressure
    "systolic_bp": (90.0, 120.0, "90–120 mmHg"),
    "diastolic_bp": (60.0, 80.0, "60–80 mmHg"),

    # Kidney function
    "creatinine": (0.6, 1.2, "0.6–1.2 mg/dL"),
    "urea": (7.0, 20.0, "7–20 mg/dL"),

    # Complete Blood Count
    "wbc": (4000.0, 11000.0, "4,000–11,000 cells/μL"),
    "rbc": (4.5, 5.5, "4.5–5.5 million/μL"),
    "platelets": (150.0, 400.0, "150,000–400,000 /μL"),

    # Thyroid
    "tsh": (0.4, 4.0, "0.4–4.0 mIU/L"),

    # Diabetes marker
    "hba1c": (None, 5.7, "< 5.7%"),
}

STATUS_CONFIG = {
    "Low":     ("🔵", "#3B82F6", 1),   # blue
    "Normal":  ("✅", "#22C55E", 0),   # green
    "High":    ("🔴", "#EF4444", 1),   # red
    "Unknown": ("⚪", "#9CA3AF", 0),   # gray
}

# Special flags for particularly dangerous values
CRITICAL_HIGH = {
    "glucose": 300,
    "cholesterol_total": 300,
    "systolic_bp": 180,
    "diastolic_bp": 120,
    "hba1c": 9.0,
    "triglycerides": 500,
}
CRITICAL_LOW = {
    "hemoglobin": 8.0,
    "glucose": 50,
    "platelets": 50,
}


def classify_parameter(key: str, param: MedicalParameter) -> AnalyzedParameter:
    """
    Apply reference-range rules to a single medical parameter.
    """
    ranges = REFERENCE_RANGES.get(key)
    value = param.value

    if ranges is None or value is None:
        icon, color, severity = STATUS_CONFIG["Unknown"]
        return AnalyzedParameter(
            name=param.name, value=value or 0, unit=param.unit,
            status="Unknown", normal_range="N/A",
            color=color, icon=icon, severity=severity,
        )

    low_thresh, high_thresh, range_text = ranges

    if low_thresh is not None and value < low_thresh:
        status = "Low"
    elif high_thresh is not None and value > high_thresh:
        status = "High"
    else:
        status = "Normal"

    icon, color, severity = STATUS_CONFIG[status]

    # Escalate severity for critical values
    if status == "High" and key in CRITICAL_HIGH and value >= CRITICAL_HIGH[key]:
        severity = 2
        color = "#B91C1C"  # darker red for critical
    if status == "Low" and key in CRITICAL_LOW and value <= CRITICAL_LOW[key]:
        severity = 2
        color = "#1D4ED8"  # darker blue for critical

    return AnalyzedParameter(
        name=param.name, value=value, unit=param.unit,
        status=status, normal_range=range_text,
        color=color, icon=icon, severity=severity,
    )


def analyze_all(params: dict[str, MedicalParameter]) -> dict[str, AnalyzedParameter]:
    """
    Classify all extracted parameters.

    Returns:
        Dict mapping parameter key → AnalyzedParameter,
        sorted so abnormal values appear first.
    """
    results = {}
    for key, param in params.items():
        results[key] = classify_parameter(key, param)

    # Sort: High-severity first, then Low, then Normal, then Unknown
    priority = {"High": 0, "Low": 1, "Normal": 2, "Unknown": 3}
    sorted_results = dict(
        sorted(
            results.items(),
            key=lambda item: (priority.get(item[1].status, 3), -item[1].severity),
        )
    )
    return sorted_results


def build_summary(analyzed: dict[str, AnalyzedParameter]) -> dict:
    """
    Build a top-level summary dict for the UI banner.
    """
    total = len(analyzed)
    abnormal = [v for v in analyzed.values() if v.status in ("High", "Low")]
    critical = [v for v in abnormal if v.severity == 2]

    return {
        "total": total,
        "normal_count": total - len(abnormal),
        "abnormal_count": len(abnormal),
        "critical_count": len(critical),
        "abnormal_names": [v.name for v in abnormal],
    }
