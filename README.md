# 🤖 InsightAI — AI Resume Job Matching System

> A premium, enterprise-grade AI-powered ATS (Applicant Tracking System) and Resume Screening Platform.  
> Serve the single-page application locally or deploy serverless to Vercel, matching candidate resumes against job descriptions with dual-provider AI support (Google Gemini + OpenAI GPT).

**Live Deployment URL:** [https://ai-resume-screening-job-match-syste-one.vercel.app](https://ai-resume-screening-job-match-syste-one.vercel.app)

---

## 📸 Platform Screenshots

<div align="center">
  <h3>Recruiter Command Console</h3>
  <img src="assets\screenshots\Recruiter Command Console.png" alt="Recruiter Command Console" width="800px" style="border-radius: 12px; margin-bottom: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);" />

  <h3>Candidate Ingestion Workspace</h3>
  <img src="assets\screenshots\Candidate Ingestion Workspace.png" alt="Candidate Ingestion Workspace" width="800px" style="border-radius: 12px; margin-bottom: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);" />

  <h3>Interactive Match Analytics & Radar Chart</h3>
  <img src="assets\screenshots\Interactive Match Analytics & Radar Chart.png" alt="Interactive Match Analytics" width="800px" style="border-radius: 12px; margin-bottom: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);" />
</div>

## 🎨 Latest UI/UX, PDF Engine, & Authentication Upgrades

We have upgraded the platform to enterprise-grade SaaS standards, resolving visual polish, PDF layout quality, authentication reliability, and mobile responsiveness:

* **Rebuilt PDF Report Generation Engine**: Restructured the printed layout format to match Greenhouse/Lever ATS platform standards:
  * *A4 Printing Geometry*: Enforced A4 boundary clamping (`297mm` height) with compact print paddings to guarantee zero blank pages.
  * *Recruiter Decision Snapshot*: Integrated a premium executive card layout summarizing fit verdict (HIRE / CONSIDER / HOLD / REJECT), score details, and candidate strengths/gaps at the very top of Page 1.
  * *Fixed Column-Width Tables*: Applied fixed table-layout width distributions and word wrapping rules on the Resume Evidence Matrix to prevent text compression and clipping.
  * *Diagnostic Watermarks & Footers*: Embedded absolute confidential footers (with dynamic page numbers) and a styled background `InsightAI` watermark.
* **Environment-Aware Global API Routing**: A unified frontend `window.fetch` interceptor automatically adjusts routing paths:
  * For local file opens (`file:///`) or custom local development server ports, it redirects requests to the local FastAPI backend (`http://127.0.0.1:8000`).
  * For production/Vercel hostnames, it preserves relative routes so that Vercel serverless routing directs traffic to the hosted python functions.
  * Eliminates mixed-content security prompts and HTTPS-to-HTTP connection warnings.
* **JWT Authentication & SSO Persistence**: Persists session tokens (`authToken`) to client-side storage across credentials forms and corporate SSO login syncs, automatically attaching them to the `Authorization: Bearer` headers of backend endpoints.
* **Premium Login Portal**: Dynamic background particle engine (`initLoginParticles()`), credentials caching (Remember Me), role-based profile themes, and multi-step onboarding wizard.
* **Resume Navigation & Workspace**: Matched skill highlighter search bar, clickable evidence mapping tables, and scroll anchors targeting *Education*, *Experience*, *Projects*, and *Skills*.

---

## ✨ Enterprise Features

| Feature | Description |
|---|---|
| 🤖 **AI Co-pilot Switcher** | 6 specialized chatbot modes: **Recruiter Assistant**, **ATS Analyzer**, **Interview Generator**, **Resume Optimizer**, **Candidate Comparator**, and **Hiring Decision Assistant**. Supports message history context memory. |
| 🔄 **Regenerate & Sweep UI** | Assistant responses feature an action toolbar with **Regenerate Answer** (fetch retry), **Collapse Toggle** (compress long card logs), and **Clear History** (sweep active chat context). |
| 📄 **Interactive ATS Workspace** | Split-screen layout containing scrollable **Interactive ATS Text Parser** on the left, and **AI Parser Annotations** + **Recruiter Comments** sidebar on the right. |
| 🆚 **Split Screen Compare** | Split-screen visual overlay showing Job Description (left), Match Insights & Gaps (center), and Highlighted Resume Text (right). |
| 🛡️ **Production-Grade Auth** | JWT-HMAC secure HTTPOnly session cookie validation, global route protection, role-based glow theming (Recruiter, HR, Admin, Candidate), dropdown navigation, profile page `/profile`, and auto-timeout. |
| 🛡️ **Explainability Mapping Matrix** | Dynamic table displaying JD requirements, resume evidence context snippet, dynamic confidence level bands, transferable skill matching, decision verdict (Match, Transferable, Missing), and detailed semantic reasoning. |
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
├── assets/
│   └── screenshots/        # Application UI screenshots (home, upload, result)
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
│   └── pipeline.py         # End-to-end pipeline orchestrator
│
├── data/
│   ├── skills.txt          # Weighted skills dictionary
│   └── job_roles.csv       # Training data: text, label
│
├── index.html              # Upgraded responsive SPA UI Dashboard
├── requirements.txt        # Python dependencies
├── README.md
├── LICENSE
└── vercel.json             # Vercel Serverless configuration
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/DurgaPavan0923/AI-Resume-Screening-Job-Match-System.git
cd AI-Resume-Screening-Job-Match-System
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
The FastAPI backend serves the HTML5 SPA dashboard directly at the root URL.

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