# 🩺 AI Medical Report Analyzer

A full-stack web application that extracts, analyzes, and explains medical lab reports using OCR and a **free, open-source LLM** — no paid API key required.

---

## 🌟 Features

| Feature | Description |
|---|---|
| 📄 **Smart OCR** | Extracts text from PDFs (pdfplumber/PyMuPDF) and images (pytesseract) |
| 🧪 **Parameter Detection** | Identifies 15+ medical values via intelligent regex parsing |
| 📊 **Rule-Based Classification** | Flags values as **High / Low / Normal** against medical reference ranges |
| 🔴 **Abnormal Highlighting** | Abnormal values highlighted in red/blue |
| 🤖 **AI Explanation** | Free LLM explains your report in plain, jargon-free English |
| 💬 **Chat with Report** | Ask follow-up questions about your specific results |
| 📋 **Summary Dashboard** | At-a-glance count of normal, abnormal, and critical values |

---

## 🆓 Free AI Options (No Anthropic / OpenAI key needed)

### Option A — Ollama (Recommended: 100% local & private)
Runs an LLM entirely on your machine. No internet needed after setup.

```bash
# 1. Install Ollama
# Visit https://ollama.com and download for your OS

# 2. Pull a model (choose one)
ollama pull llama3          # Best quality (~4.7 GB)
ollama pull mistral         # Fast & good (~4.1 GB)
ollama pull phi3            # Lightweight (~2.3 GB)

# 3. Set in .env
AI_PROVIDER=ollama
OLLAMA_MODEL=llama3
```

### Option B — Groq (Free cloud API, very fast)
Free tier with generous limits. No credit card required.

```bash
# 1. Sign up free at https://console.groq.com
# 2. Create an API key (free)
# 3. Set in .env
AI_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
```

---

## 🗂️ Project Structure

```
ai-medical-analyzer/
│
├── app.py                  # Main Streamlit UI
├── backend.py              # Processing pipeline (OCR → Parse → Analyze → AI)
│
├── utils/
│   ├── __init__.py
│   ├── ocr.py              # Text extraction (PDF + image)
│   ├── parser.py           # Medical parameter extraction (regex)
│   └── analyzer.py         # Rule-based classification engine
│
├── sample_report.txt       # Sample test report
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **OCR (PDF)** | pdfplumber, PyMuPDF |
| **OCR (Image)** | pytesseract + Pillow |
| **AI (local)** | Ollama (Llama3 / Mistral / Phi3) |
| **AI (cloud)** | Groq free API (Llama3) |
| **Language** | Python 3.10+ |

---

## 🚀 How to Run Locally

### 1. Prerequisites

- Python 3.10+
- **Tesseract OCR** (for image files):
  - Ubuntu: `sudo apt-get install tesseract-ocr`
  - macOS: `brew install tesseract`
  - Windows: https://github.com/tesseract-ocr/tesseract

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure AI Provider

```bash
cp .env.example .env
# Edit .env — choose ollama or groq
```

### 4. (If using Ollama) Start Ollama

```bash
ollama serve          # start the server
ollama pull llama3    # download the model (first time only)
```

### 5. Run the App

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

---

## 🧪 Testing with Sample Data

Use `sample_report.txt` — it contains a realistic lab report with multiple abnormal values.

**Expected results:**
- 🔴 High: Glucose, HbA1c, Cholesterol, LDL, Triglycerides, Systolic BP, Diastolic BP
- 🔵 Low: Hemoglobin, HDL
- ✅ Normal: Creatinine, Blood Urea, Platelets, TSH

---

## ⚠️ Medical Disclaimer

This tool is for **informational purposes only**. It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider before making any health decisions.
