# 🤖 AI Resume Screening & Job Match System

> A premium, enterprise-grade AI-powered ATS (Applicant Tracking System) and Resume Screening Platform.  
> Serve the single-page application locally or on serverless Vercel, matching candidate resumes against job descriptions with dual-provider AI support (Google Gemini + OpenAI GPT).

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 PDF & ZIP Parsing | Extract clean text from single PDF resumes or bulk upload multiple profiles via ZIP archives. |
| 🧠 Smart JD Analyzer | Auto-analyze Job Descriptions for quality scores, minimum experience required, ideal candidate summaries, and text optimization recommendations. |
| 🛡️ Enterprise Security | CENTRALIZED server-side key management,滑动窗口 rate-limiting (throttling), CORS checks, and client PII masking (emails/phones). |
| ⚙️ Configuration Manager | Toggle active AI engines, manage API credentials with visibility switches, and monitor real-time token telemetry/billing logs. |
| 🎛️ Advanced Sourcing Filters | Candidate search bar combined with advanced Boolean search query syntax (e.g. `Python AND ML NOT Java`), experience sliders, score ranges, and education filters. |
| 🔗 Staging Pipeline | Visually advance candidates through a 6-stage recruiting pipeline (Applied → Screened → Shortlisted → Interview → Final → Hired). |
| 🧭 Candidate Drawer | Slide-over drawer loading career trajectory timelines, domain expertise breakdowns, social validations (e.g. GitHub contributions), and credentials. |
| 📝 Recruiter Notes & Ratings | Add bookmark status, comment logs, and star ratings (technical/culture fit) mapped dynamically to candidate states. |
| 🎨 Glassmorphic Theme | Stunning dark/light mode responsive bento layout built with HTML5, TailwindCSS, and Google Material Icons. |
| 🖨️ Print-Ready Layout | Clean, high-contrast `@media print` style sheets formatting report sections and hiding sidebars/widgets during print. |
| 📥 CSV Export | Download ranked candidate shortlists directly as a spreadsheet. |

---

## 🗂️ Project Structure

```
AI-Resume-Job-Match-System/
│
├── api/
│   └── main.py             # FastAPI REST API & Static Page Router
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
| `POST` | `/api/analyze` | Process PDF/ZIP resumes against a target JD |
| `POST` | `/api/analyze-jd` | Analyze JD parameters & extract quality metrics |
| `GET` | `/api/settings` | Retrieve active AI provider, security states, and telemetry |
| `POST` | `/api/settings` | Save active AI provider and server-side credentials |
| `POST` | `/api/settings/test`| Run a lightweight connection check to Gemini/OpenAI |
| `POST` | `/api/rewrite` | ATS optimization helper for resume bullet points |
| `POST` | `/api/chat` | Chat request handler for recruiting co-pilot |

---

## 🛠️ Tech Stack

* **Frontend** — HTML5, Javascript (ES6), TailwindCSS, Chart.js, Google Material Icons.
* **Backend** — FastAPI, Uvicorn server, HTTP Client (`urllib`).
* **NLP & Models** — scikit-learn TF-IDF, NLTK-based cleaning.
* **LLM Integrations** — Google Gemini (Default), OpenAI.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.