# 🤖 AI Resume Screening & Job Match System

> A premium, enterprise-grade AI-powered ATS (Applicant Tracking System) and Resume Screening Platform.  
> Serve the single-page application locally or on serverless Vercel, matching candidate resumes against job descriptions with dual-provider AI support (Google Gemini + OpenAI GPT).

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 Interactive Resume tabs | Switch between visual Document PDF view and an **Interactive ATS Text Parser** highlighting matched and missing skills with live tooltips. |
| 🆚 Side-by-Side Compare | Split-screen visual overlay showing Job Description (left), Match Insights & Gaps (center), and Highlighted Resume Text (right). |
| 🛡️ Multi-Role Auth Portal | Secure login panel supporting tailored dashboard interfaces for **Recruiters**, **HR Managers**, **Admins**, and **Candidates**. |
| 🧭 Guided Onboarding Tour | Fully interactive step-by-step onboarding walkthrough guide explaining system features to new users. |
| 📈 Candidate Strength Meter | 5-bar animated profile capability analytics (Technical, Projects, Experience Depth, Leadership, and ATS Readiness). |
| 🗺️ AI Career Roadmap | Actionable learning pathways and study path links to Coursera/Udemy to help candidates bridge identified skill gaps. |
| 👥 Backup Recommendations | Dynamic candidate pool recommendations suggesting backup and alternative candidates matching similar profiles. |
| ⚖️ Fairness & Bias Monitor | Objective blind evaluation monitor redacting PII names, genders, and locations to ensure bias-free skill-based screenings. |
| 📅 Interview Scheduling | Coordinate technical interviews via mock Google Calendar sync, generating invitation templates and Zoom meeting links. |
| 🤖 Pool-Wide AI Assistant | Recruiters can ask the co-pilot questions across the entire candidates pool (e.g. *"Who has the strongest Python skills?"*). |
| 📊 Admin Analytics | Aggregate screenings dashboard, average match scores, API error ratios, average processing times, and recruiting funnel metrics. |
| 🛡️ Enterprise Security | Server-side API key management, sliding-window rate limits, client PII masking, and strict CORS configuration switches. |

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
| `POST` | `/api/analyze` | Process PDF/ZIP resumes against a target JD and return text highlights & summaries |
| `POST` | `/api/analyze-jd` | Analyze JD parameters & extract quality metrics |
| `GET` | `/api/settings` | Retrieve active AI provider, security states, and telemetry |
| `POST` | `/api/settings` | Save active AI provider and server-side credentials |
| `POST` | `/api/settings/test`| Run a lightweight check to Gemini/OpenAI credentials |
| `POST` | `/api/rewrite` | ATS optimization helper for resume bullet points |
| `POST` | `/api/chat` | Chat request handler for a single candidate profile |
| `POST` | `/api/chat-all` | Comparative pool chat assistant handler for all active candidates |

---

## 🛠️ Tech Stack

* **Frontend** — HTML5, Javascript (ES6), TailwindCSS, Chart.js, Google Material Icons.
* **Backend** — FastAPI, Uvicorn server, HTTP Client (`urllib`).
* **NLP & Models** — scikit-learn TF-IDF, NLTK-based cleaning.
* **LLM Integrations** — Google Gemini (Default), OpenAI.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.