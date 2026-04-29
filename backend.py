"""
backend.py - Core processing pipeline.

Orchestrates: OCR → Parsing → Analysis → AI Explanation

AI Provider options (all FREE — no Anthropic key needed):
  1. Ollama  — runs LLMs locally on your machine (100% free, private)
  2. Groq    — free cloud API (very fast, generous free tier)

Set in .env:
  AI_PROVIDER=ollama        (default, no key needed)
  AI_PROVIDER=groq
  GROQ_API_KEY=gsk_...      (only if AI_PROVIDER=groq)
  OLLAMA_MODEL=llama3       (optional, default: llama3)
  OLLAMA_URL=http://localhost:11434  (optional)
"""

import os
import json
import urllib.request
import urllib.error
from utils.ocr import extract_text
from utils.parser import extract_parameters, summarize_extracted
from utils.analyzer import analyze_all, build_summary, AnalyzedParameter


# ---------------------------------------------------------------------------
# Shared prompt helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a friendly, empathetic medical report explainer.
Your job is to help patients understand their lab results in simple, reassuring language.

Rules:
- Avoid complex medical jargon. Explain any technical terms you must use.
- Be factual but compassionate. Do not alarm the patient unnecessarily.
- For abnormal values, explain what they might mean and suggest practical next steps.
- Always end with: recommend the patient discuss results with their doctor.
- Keep explanations concise (3-5 short paragraphs max).
- Never diagnose. Never prescribe. You are an explainer, not a doctor.
"""


def _build_param_summary(analyzed: dict) -> str:
    lines = []
    for key, param in analyzed.items():
        lines.append(
            f"- {param.name}: {param.value} {param.unit} -> {param.status} "
            f"(normal range: {param.normal_range})"
        )
    return "\n".join(lines) if lines else "No standard parameters detected."


def _build_explanation_prompt(raw_text: str, analyzed: dict) -> str:
    param_summary = _build_param_summary(analyzed)
    return f"""Please explain this medical report in simple, friendly language.

Detected Parameters:
{param_summary}

Raw Report Text (for context):
{raw_text[:2000]}

Write a clear summary that:
1. Starts with an overall health snapshot (1-2 sentences)
2. Explains each abnormal value in plain English
3. Notes the values that are healthy (briefly)
4. Gives 2-3 practical lifestyle suggestions
5. Ends with a reminder to consult their doctor
"""


def _build_chat_prompt(analyzed: dict, user_question: str) -> str:
    param_summary = _build_param_summary(analyzed)
    return f"""The patient is asking about their medical report.

Report parameters:
{param_summary}

Patient question: {user_question}

Answer in simple, friendly language. Reference specific values when relevant."""


# ---------------------------------------------------------------------------
# Provider 1: Ollama (local, completely free)
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, chat_history=None) -> str:
    """
    Call a locally running Ollama instance.
    Install: https://ollama.com  then run: ollama pull llama3
    """
    base_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model    = os.environ.get("OLLAMA_MODEL", "llama3")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["message"]["content"].strip()
    except urllib.error.URLError as e:
        return (
            "**Ollama not reachable** - is it running?\n\n"
            "Steps to fix:\n"
            "1. Install Ollama: https://ollama.com\n"
            "2. Run: ollama serve\n"
            "3. Pull a model: ollama pull llama3\n\n"
            "Or switch to Groq by setting AI_PROVIDER=groq in your .env\n\n"
            f"(Error: {e})"
        )
    except Exception as e:
        return f"Ollama error: {str(e)}"


# ---------------------------------------------------------------------------
# Provider 2: Groq (free cloud API — very fast)
# ---------------------------------------------------------------------------

def _call_groq(prompt: str, chat_history=None) -> str:
    """
    Call Groq free API. Get a free key at https://console.groq.com
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return (
            "Groq API key missing - set GROQ_API_KEY in your .env file.\n"
            "Get a free key at: https://console.groq.com\n"
            "Or use local Ollama: set AI_PROVIDER=ollama"
        )

    model = os.environ.get("GROQ_MODEL", "llama3-8b-8192")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.7,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        if e.code == 401:
            return "Invalid Groq API key. Check your GROQ_API_KEY in .env"
        if e.code == 429:
            return "Groq rate limit reached. Wait a moment and try again."
        return f"Groq API error {e.code}: {body[:300]}"
    except Exception as e:
        return f"Groq error: {str(e)}"


# ---------------------------------------------------------------------------
# Main AI dispatcher
# ---------------------------------------------------------------------------

def generate_explanation(
    raw_text: str,
    analyzed: dict,
    chat_history=None,
    user_question: str = None,
) -> str:
    """
    Generate a plain-English explanation using the configured AI provider.
    Select provider via AI_PROVIDER in .env: 'ollama' or 'groq'
    """
    provider = os.environ.get("AI_PROVIDER", "ollama").lower().strip()

    if user_question:
        prompt = _build_chat_prompt(analyzed, user_question)
    else:
        prompt = _build_explanation_prompt(raw_text, analyzed)

    if provider == "groq":
        return _call_groq(prompt, chat_history)
    elif provider == "ollama":
        return _call_ollama(prompt, chat_history)
    else:
        return (
            f"Unknown AI_PROVIDER '{provider}'. "
            "Set AI_PROVIDER=ollama or AI_PROVIDER=groq in your .env"
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_report(file_bytes: bytes, filename: str) -> dict:
    """
    Full processing pipeline: OCR -> Parse -> Analyze -> AI Explanation
    """
    result = {
        "raw_text": "",
        "params": {},
        "analyzed": {},
        "summary": {},
        "explanation": "",
        "error": None,
    }

    # Step 1: OCR
    raw_text = extract_text(file_bytes, filename)
    if raw_text.startswith("[ERROR]") or raw_text.startswith("[OCR ERROR]"):
        result["error"] = raw_text
        return result
    result["raw_text"] = raw_text

    # Step 2: Parse
    params = extract_parameters(raw_text)
    result["params"] = params

    # Step 3: Classify
    analyzed = analyze_all(params)
    result["analyzed"] = analyzed
    result["summary"] = build_summary(analyzed)

    # Step 4: AI Explanation
    explanation = generate_explanation(raw_text, analyzed)
    result["explanation"] = explanation

    return result
