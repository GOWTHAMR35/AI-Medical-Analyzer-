"""
app.py - AI Medical Report Analyzer
Main Streamlit application.

Run with:  streamlit run app.py
"""

import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from .env file
load_dotenv()

from backend import process_report, generate_explanation

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Medical Report Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — clean medical aesthetic
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

  html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
  }

  /* Header */
  .app-header {
      background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 60%, #0F4C81 100%);
      border-radius: 16px;
      padding: 36px 40px;
      margin-bottom: 28px;
      color: white;
      position: relative;
      overflow: hidden;
  }
  .app-header::before {
      content: '';
      position: absolute;
      top: -40px; right: -40px;
      width: 220px; height: 220px;
      border-radius: 50%;
      background: rgba(255,255,255,0.04);
  }
  .app-header h1 {
      font-family: 'DM Serif Display', serif;
      font-size: 2.2rem;
      margin: 0 0 6px 0;
      color: white;
  }
  .app-header p {
      opacity: 0.72;
      font-size: 1rem;
      margin: 0;
  }

  /* Summary cards */
  .summary-card {
      background: white;
      border-radius: 12px;
      padding: 18px 22px;
      text-align: center;
      box-shadow: 0 2px 12px rgba(0,0,0,0.06);
      border-top: 4px solid #3B82F6;
  }
  .summary-card.abnormal { border-top-color: #EF4444; }
  .summary-card.normal   { border-top-color: #22C55E; }
  .summary-card.critical { border-top-color: #7C2D12; }

  .summary-card .count {
      font-size: 2.2rem;
      font-weight: 700;
      line-height: 1.1;
      color: #0F172A;
  }
  .summary-card .label {
      font-size: 0.78rem;
      color: #64748B;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-top: 4px;
  }

  /* Parameter row */
  .param-row {
      display: flex;
      align-items: center;
      padding: 12px 16px;
      border-radius: 10px;
      margin-bottom: 8px;
      background: #F8FAFC;
      border-left: 4px solid #E2E8F0;
      transition: transform 0.15s;
  }
  .param-row:hover { transform: translateX(3px); }
  .param-row.high  { border-left-color: #EF4444; background: #FFF5F5; }
  .param-row.low   { border-left-color: #3B82F6; background: #EFF6FF; }
  .param-row.normal{ border-left-color: #22C55E; background: #F0FDF4; }

  .param-name  { font-weight: 600; font-size: 0.92rem; color: #1E293B; flex: 2; }
  .param-val   { font-size: 1.05rem; font-weight: 700; color: #0F172A; flex: 1; text-align: center; }
  .param-range { font-size: 0.78rem; color: #64748B; flex: 2; text-align: center; }
  .param-badge {
      flex: 1; text-align: right;
      font-size: 0.75rem; font-weight: 600;
      padding: 3px 10px; border-radius: 20px;
  }
  .badge-high   { background: #FEE2E2; color: #B91C1C; }
  .badge-low    { background: #DBEAFE; color: #1D4ED8; }
  .badge-normal { background: #DCFCE7; color: #15803D; }

  /* Section headers */
  .section-title {
      font-family: 'DM Serif Display', serif;
      font-size: 1.35rem;
      color: #0F172A;
      padding-bottom: 6px;
      border-bottom: 2px solid #E2E8F0;
      margin-bottom: 16px;
  }

  /* AI explanation box */
  .ai-box {
      background: linear-gradient(135deg, #F0F9FF, #EFF6FF);
      border-radius: 14px;
      padding: 24px 28px;
      border: 1px solid #BAE6FD;
      line-height: 1.7;
      font-size: 0.95rem;
  }

  /* Chat bubble */
  .chat-user {
      background: #1E40AF;
      color: white;
      border-radius: 16px 16px 4px 16px;
      padding: 10px 16px;
      margin: 6px 0;
      max-width: 80%;
      margin-left: auto;
      font-size: 0.9rem;
  }
  .chat-ai {
      background: #F1F5F9;
      color: #1E293B;
      border-radius: 16px 16px 16px 4px;
      padding: 10px 16px;
      margin: 6px 0;
      max-width: 85%;
      font-size: 0.9rem;
      line-height: 1.6;
  }

  /* Upload area */
  .upload-hint {
      background: #F8FAFC;
      border: 2px dashed #CBD5E1;
      border-radius: 12px;
      padding: 20px;
      text-align: center;
      color: #64748B;
      font-size: 0.88rem;
      margin-top: 8px;
  }

  /* Hide Streamlit default elements */
  #MainMenu { visibility: hidden; }
  footer    { visibility: hidden; }
  .stDeployButton { display: none; }

  /* Disclaimer */
  .disclaimer {
      background: #FFFBEB;
      border: 1px solid #FCD34D;
      border-radius: 10px;
      padding: 12px 16px;
      font-size: 0.8rem;
      color: #78350F;
      margin-top: 20px;
  }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "result": None,
        "chat_history": [],      # [{role, content}, ...]
        "chat_display": [],      # [(user_msg, ai_msg), ...]
        "processing": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="app-header">
  <h1>🩺 AI Medical Report Analyzer</h1>
  <p>Upload a lab report · Instantly extract values · Get plain-English explanations</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Section 1 — File Upload
# ---------------------------------------------------------------------------

st.markdown('<div class="section-title">📂 Upload Your Report</div>', unsafe_allow_html=True)

col_upload, col_hint = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Choose a PDF or image file",
        type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff"],
        label_visibility="collapsed",
    )

with col_hint:
    st.markdown("""
    <div class="upload-hint">
      <b>Accepted formats</b><br>
      PDF &nbsp;·&nbsp; PNG &nbsp;·&nbsp; JPG &nbsp;·&nbsp; JPEG<br>
      BMP &nbsp;·&nbsp; TIFF<br><br>
      <i>Your file is processed locally and never stored.</i>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file:
    analyze_btn = st.button("🔬 Analyze Report", type="primary", use_container_width=True)

    if analyze_btn:
        st.session_state.chat_history = []
        st.session_state.chat_display = []

        with st.spinner("Extracting text and analyzing values…"):
            file_bytes = uploaded_file.read()
            result = process_report(file_bytes, uploaded_file.name)
            st.session_state.result = result

        if result.get("error"):
            st.error(result["error"])
        else:
            st.success(f"✅ Report analyzed! Found **{result['summary']['total']}** parameters.")


# ---------------------------------------------------------------------------
# Results section — only shown after successful processing
# ---------------------------------------------------------------------------

result = st.session_state.get("result")

if result and not result.get("error"):
    summary = result["summary"]
    analyzed = result["analyzed"]

    st.divider()

    # ── Summary banner ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Summary</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="summary-card">
          <div class="count">{summary['total']}</div>
          <div class="label">Parameters Found</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="summary-card normal">
          <div class="count">{summary['normal_count']}</div>
          <div class="label">Normal</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="summary-card abnormal">
          <div class="count">{summary['abnormal_count']}</div>
          <div class="label">Abnormal</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="summary-card critical">
          <div class="count">{summary['critical_count']}</div>
          <div class="label">Critical</div>
        </div>""", unsafe_allow_html=True)

    if summary["abnormal_names"]:
        st.warning(f"⚠️ Abnormal values detected: **{', '.join(summary['abnormal_names'])}**")

    st.divider()

    # ── Parameter analysis table ─────────────────────────────────────────────
    st.markdown('<div class="section-title">🧪 Detailed Analysis</div>', unsafe_allow_html=True)

    # Build HTML rows
    rows_html = ""
    for key, param in analyzed.items():
        status_lower = param.status.lower()
        css_class = status_lower if status_lower in ("high", "low", "normal") else ""
        badge_class = f"badge-{status_lower}" if status_lower in ("high", "low", "normal") else ""
        rows_html += f"""
        <div class="param-row {css_class}">
          <span class="param-name">{param.icon} {param.name}</span>
          <span class="param-val">{param.value} <small style="font-weight:400;color:#64748B">{param.unit}</small></span>
          <span class="param-range">{param.normal_range}</span>
          <span class="param-badge {badge_class}">{param.status}</span>
        </div>"""

    st.markdown(rows_html, unsafe_allow_html=True)

    st.divider()

    # ── AI Explanation ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🤖 AI Explanation</div>', unsafe_allow_html=True)

    explanation = result.get("explanation", "")
    if explanation:
        st.markdown(f'<div class="ai-box">{explanation}</div>', unsafe_allow_html=True)
    else:
        st.info("Generating explanation…")

    st.divider()

    # ── Chat with Report ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">💬 Chat with Your Report</div>', unsafe_allow_html=True)
    st.caption("Ask follow-up questions about your results in plain language.")

    # Display chat history
    for user_msg, ai_msg in st.session_state.chat_display:
        st.markdown(f'<div class="chat-user">🙋 {user_msg}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-ai">🤖 {ai_msg}</div>', unsafe_allow_html=True)

    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        user_q = st.text_input(
            "Your question",
            placeholder="e.g. Is my glucose level dangerous? What should I eat?",
            label_visibility="collapsed",
        )
        send_btn = st.form_submit_button("Send →", use_container_width=True)

    if send_btn and user_q.strip():
        with st.spinner("Thinking…"):
            # Build history for the API
            history_for_api = []
            for u, a in st.session_state.chat_display:
                history_for_api.append({"role": "user", "content": u})
                history_for_api.append({"role": "assistant", "content": a})

            ai_response = generate_explanation(
                raw_text=result["raw_text"],
                analyzed=analyzed,
                chat_history=history_for_api,
                user_question=user_q.strip(),
            )

            # Store in session
            st.session_state.chat_display.append((user_q.strip(), ai_response))
            history_for_api.append({"role": "user", "content": user_q.strip()})
            history_for_api.append({"role": "assistant", "content": ai_response})
            st.session_state.chat_history = history_for_api

        st.rerun()

    st.divider()

    # ── Raw Extracted Text (collapsible) ─────────────────────────────────────
    with st.expander("📄 View Extracted Raw Text"):
        st.text_area(
            "OCR Output",
            value=result["raw_text"],
            height=240,
            label_visibility="collapsed",
        )

    # ── Disclaimer ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
      ⚠️ <strong>Medical Disclaimer:</strong> This tool is for informational purposes only and does not
      constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider
      regarding your medical results and health decisions.
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Empty state — no file uploaded yet
# ---------------------------------------------------------------------------

elif not uploaded_file:
    st.markdown("""
    <div style="text-align:center; padding: 40px 20px; color: #94A3B8;">
      <div style="font-size: 4rem; margin-bottom: 16px;">🔬</div>
      <div style="font-size: 1.1rem; font-weight: 600; color: #475569; margin-bottom: 8px;">
        Upload a medical report to get started
      </div>
      <div style="font-size: 0.88rem;">
        Supports PDF lab reports and image scans.<br>
        Works with blood tests, lipid panels, CBC reports, thyroid panels, and more.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature overview
    st.divider()
    fc1, fc2, fc3, fc4 = st.columns(4)
    features = [
        ("📄", "Smart OCR", "Extracts text from PDFs and scanned images automatically"),
        ("🧪", "Parameter Detection", "Identifies glucose, hemoglobin, cholesterol, BP & more"),
        ("📊", "Instant Classification", "Flags High / Low / Normal values against medical norms"),
        ("🤖", "AI Explanation", "Explains your report in simple, jargon-free language"),
    ]
    for col, (icon, title, desc) in zip([fc1, fc2, fc3, fc4], features):
        with col:
            st.markdown(f"""
            <div style="background:#F8FAFC; border-radius:12px; padding:20px; text-align:center; height:140px; border:1px solid #E2E8F0;">
              <div style="font-size:2rem">{icon}</div>
              <div style="font-weight:600; font-size:0.9rem; color:#1E293B; margin:8px 0 4px">{title}</div>
              <div style="font-size:0.78rem; color:#64748B">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
