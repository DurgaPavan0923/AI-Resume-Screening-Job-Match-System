# 🤖 InsightAI — AI-Native Recruiting Operating System

> A premium, enterprise-grade AI-powered ATS (Applicant Tracking System) and Resume Screening Platform.  
> Serve the single-page application locally or deploy serverless to Vercel, matching candidate resumes against job descriptions with dual-provider AI support (Google Gemini + OpenAI GPT).

**Live Deployment URL:** [https://ai-resume-screening-job-match-syste-one.vercel.app](https://ai-resume-screening-job-match-syste-one.vercel.app)

---

## ✨ Enterprise Features

| Feature | Description |
|---|---|
| 🤖 **AI Co-pilot Switcher** | 4 specialized chatbot modes: **Recruiter Mode** (fit & qualifications), **ATS Analyzer** (keywords & gaps), **Interview Generator** (custom technical/HR questions), and **Resume Optimizer** (impact rewrites). |
| 🔄 **Regenerate & Collapse UI** | Assistant responses feature an action toolbar with **Regenerate Answer** (fetch retry) and **Collapse Toggle** (compress long card logs down to `max-h-14` dynamically). |
| 📄 **Interactive Match Highlights** | Switch between visual Document PDF view and an **Interactive ATS Text Parser** highlighting matched and missing skills with live tooltips. |
| 🆚 **Split Screen Compare** | Split-screen visual overlay showing Job Description (left), Match Insights & Gaps (center), and Highlighted Resume Text (right). |
| 🛡️ **Multi-Role Auth Portal** | Secure login panel supporting tailored dashboard interfaces for **Recruiters**, **HR Managers**, **Admins**, and **Candidates**. |
| 🗺️ **Skills Knowledge Graph** | Beautiful SVG-based interactive dependency tree depicting candidate skills, clusters, and overlap maps. |
| 🎙️ **Voice AI Interview Modal** | Select focus areas and start a behavioral or technical mock interview with pulsing soundwaves, live transcripts, and speech fluency audits (WPM, sentiment). |
| 🛡️ **Resume Fraud signals** | Scans for keyword stuffing index, candidate experience anomalies, and displays a mock GitHub contribution matrix. |
| ⏳ **Multi-Agent Pipeline** | Loading interface detailing real-time agent statuses (Ingestion, Scoring, Trajectory, Verification, Verdict). |
| 📈 **Candidate Radar Analytics** | Dynamic 6-axis polygon mapping candidate alignment across semantic similarity, experience, keywords, and structural baselines. |
| ⚖️ **GDPR & PII Blind Masking** | GDPR data purging, PII redactor (anonymizing names/emails), and local Offline AI simulator switches. |
| 🤝 **Collaboration Board** | Star ratings, reviewer feedback comments log, approvals history, and automated shortlisted candidates follow-up email generator. |

---

## 🗂️ Project Structure

```
AI-Resume-Job-Match-System/
│
├── api/
│   └── main.py             # FastAPI REST API, AI Modes Router & Static Server
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py       # Text cleaning & normalisation
│   ├── pdf_parser.py       # PDF → plain text extraction
│   ├── skill_extractor.py  # Skill detection, synonyms, & weighting
│   ├── similarity.py       # TF-IDF / Cosine similarity matching
│   ├── job_predictor.py    # ML role classifier model
│   ├── experience_extractor.py
│   ├── education_parser.py
│   ├── pipeline.py         # End-to-end pipeline orchestrator
│   └── train.py            # Model training & vectorizer
│
├── data/
│   ├── skills.txt          # Weighted skills dictionary
│   └── job_roles.csv       # Training data: text, label
│
├── index.html              # Upgraded responsive SPA UI Dashboard
├── styles.css              # Custom styled definitions
├── requirements.txt        # Python dependencies
├── README.md
├── LICENSE
└── vercel.json             # Vercel Serverless configuration
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/AI-Resume-Job-Match-System.git
cd AI-Resume-Job-Match-System
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI server locally

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Access the Platform

Open your browser and navigate to:
```
http://localhost:8000/
```
The FastAPI backend serves the premium HTML5 SPA dashboard directly at the root URL.

---

## ⚙️ AI Models & Credentials Configuration

The system uses **Google Gemini** (`gemini-2.5-flash`) as the default active AI provider for out-of-the-box resume analysis and JD parsing. 
* To update your API credentials or switch the active provider to **OpenAI GPT** (`gpt-4o-mini`), click the **Settings** link in the navigation bar.
* Test your API connection directly from the UI and save credentials securely on the server-side.

---

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the SPA frontend dashboard |
| `POST` | `/api/analyze` | Process PDF/ZIP resumes against a target JD and return text highlights & summaries |
| `POST` | `/api/analyze-jd` | Analyze JD parameters & extract quality metrics |
| `GET` | `/api/settings` | Retrieve active AI provider, security states, and telemetry |
| `POST` | `/api/settings` | Save active AI provider and server-side credentials |
| `POST` | `/api/settings/test`| Run a lightweight check to Gemini/OpenAI credentials |
| `POST` | `/api/rewrite` | ATS optimization helper for resume bullet points |
| `POST` | `/api/chat` | Chat request handler for single candidates supporting AI Modes and explicit prompt actions |
| `POST` | `/api/chat-all` | Comparative pool chat assistant handler for all active candidates |
| `POST` | `/api/generate-jd` | Auto-generate and optimize job descriptions for ATS compliance and keyword alignment |
| `POST` | `/api/mock-interview` | Start or resume a simulated voice AI technical or behavioral interview, returning fluency & speech metrics |

---

## 🛠️ Tech Stack

* **Frontend** — HTML5, Javascript (ES6), TailwindCSS, Chart.js, Google Material Icons.
* **Backend** — FastAPI, Uvicorn server, HTTP Client (`urllib`).
* **NLP & Models** — scikit-learn TF-IDF, NLTK-based cleaning.
* **LLM Integrations** — Google Gemini (Default), OpenAI.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.