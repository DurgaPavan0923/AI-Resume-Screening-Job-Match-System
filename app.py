import base64

import pandas as pd
import plotly.express as px
import streamlit as st

from config import SKILLS_PATH
from src.education_parser import extract_education
from src.experience_extractor import extract_experience
from src.highlighter import highlight_text  # noqa: F401  (available for future use)
from src.job_predictor import predict_roles
from src.pdf_parser import parse_pdf
from src.preprocess import clean_text
from src.similarity import compute_similarity
from src.skill_extractor import extract_skills, load_skills
from src.train import train_model
from utils.helpers import format_skills, validate_input

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screening Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────
# EXTERNAL CSS
# ──────────────────────────────────────────────────────────────
def local_css(file_name: str) -> None:
    """Inject an external CSS file into the Streamlit app."""
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("assets/styles.css")


# ──────────────────────────────────────────────────────────────
# CACHED MODEL
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model…")
def get_model():
    return train_model()


model, vectorizer = get_model()


# ──────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────
def skill_match_score(
    jd_skills: dict,
    resume_skills: dict,
    skills_db: dict,
) -> float:
    """Weighted skill-match ratio (0–1)."""
    if not jd_skills:
        return 0.0
    total = sum(skills_db.get(s, 1) for s in jd_skills)
    matched = sum(skills_db.get(s, 1) for s in resume_skills if s in jd_skills)
    return matched / total if total else 0.0


def get_decision(score: float) -> str:
    """Return an HTML hiring-decision badge based on the score."""
    if score >= 75:
        return "<span style='color:#1de9b6;font-weight:700'>🟢 Hire</span>"
    elif score >= 50:
        return "<span style='color:#ffb300;font-weight:700'>🟡 Consider</span>"
    return "<span style='color:#ff5252;font-weight:700'>🔴 Reject</span>"


def skill_gap(jd_skills: dict, resume_skills: dict) -> list:
    """Return skills present in the JD but missing from the resume."""
    jd_set = set(jd_skills.keys() if isinstance(jd_skills, dict) else jd_skills)
    resume_set = set(resume_skills.keys() if isinstance(resume_skills, dict) else resume_skills)
    return sorted(jd_set - resume_set)


def generate_role_based_questions(role: str, skills) -> list:
    """Generate tailored interview questions for a given role and skill set."""
    role = str(role).lower()
    skill_list = list(skills.keys()) if isinstance(skills, dict) else list(skills)

    questions: list[str] = []

    if "data scientist" in role:
        questions += [
            "Explain supervised vs unsupervised learning with real-world examples.",
            "How do you handle missing data in a dataset?",
            "What evaluation metrics do you use and when?",
        ]
    elif "ai engineer" in role:
        questions += [
            "Explain how neural networks learn from data.",
            "What is backpropagation and why does it matter?",
            "How do you optimise a deep learning model for production?",
        ]
    elif "backend" in role:
        questions += [
            "What principles guide your REST API design?",
            "How do you implement secure authentication and authorisation?",
            "How have you scaled backend systems under high load?",
        ]
    elif "frontend" in role or "web" in role:
        questions += [
            "How do you approach responsive design?",
            "Explain the browser's rendering pipeline.",
            "How do you optimise frontend performance?",
        ]
    else:
        questions += [
            "Walk me through a system you designed end-to-end.",
            "How do you prioritise tasks when deadlines conflict?",
        ]

    for skill in skill_list[:5]:
        questions.append(f"What is your depth of experience with {skill}?")

    questions += [
        "Describe the most technically challenging project you have worked on.",
        "How do you approach debugging a problem you have never seen before?",
        "Why are you the right person for this role?",
    ]

    return questions


def pdf_iframe(file_obj) -> str:
    """Return an HTML iframe string for embedding a PDF."""
    file_obj.seek(0)
    b64 = base64.b64encode(file_obj.read()).decode("utf-8")
    return (
        f'<iframe src="data:application/pdf;base64,{b64}" '
        f'width="100%" height="520px" '
        f'style="border:none;border-radius:12px;"></iframe>'
    )


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 AI Resume Screener")
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown(
        """
        1. Paste the **Job Description** on the right  
        2. Upload one or more **PDF resumes**  
        3. Click **Analyse Candidates**  
        4. Review ranked results & export  
        """
    )
    st.markdown("---")
    st.markdown("### Tips")
    st.write("✔ Use detailed job descriptions")
    st.write("✔ List required skills explicitly")
    st.write("✔ Upload multiple resumes for ranking")
    st.markdown("---")
    st.caption("Advanced AI Resume Screening System")


# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────
st.markdown(
    '<h1 class="main-title">AI Resume Screening Dashboard</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">Smart hiring powered by AI — rank, analyse, and shortlist with confidence</p>',
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# INPUT PANEL
# ──────────────────────────────────────────────────────────────
col_jd, col_up = st.columns([2, 1])

with col_jd:
    job_desc = st.text_area(
        "Job Description",
        placeholder="Paste the full job description here…",
        height=220,
    )

with col_up:
    files = st.file_uploader(
        "Upload Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDF resumes at once.",
    )

st.markdown("<hr>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# ANALYSE BUTTON
# ──────────────────────────────────────────────────────────────
run_analysis = st.button("🔍 Analyse Candidates", use_container_width=True)

if run_analysis:

    # ── Validation ────────────────────────────────────────────
    valid, msg = validate_input(job_desc, files)
    if not valid:
        st.warning(msg)
        st.stop()

    # ── Core processing ───────────────────────────────────────
    skills_db = load_skills(SKILLS_PATH)
    job_clean = clean_text(job_desc)
    jd_skills = extract_skills(job_clean, skills_db) or {}

    results: list[dict] = []
    raw_texts: dict = {}

    progress_bar = st.progress(0, text="Processing resumes…")

    with st.spinner("Analysing resumes — please wait…"):
        for idx, file in enumerate(files):
            try:
                text = parse_pdf(file)

                if not text.strip():
                    st.warning(f"⚠️ Could not extract text from **{file.name}** — skipping.")
                    continue

                # Cache raw file bytes for PDF preview
                file.seek(0)
                raw_texts[file.name] = file

                clean = clean_text(text)

                similarity_score = compute_similarity(job_clean, clean, vectorizer)

                skills = extract_skills(clean, skills_db) or {}
                s_score = skill_match_score(jd_skills, skills, skills_db)

                experience = extract_experience(clean)
                education = extract_education(clean)

                roles, ml_roles = predict_roles(clean, skills, model, vectorizer)
                role_display = ", ".join(roles) if roles else "Unclassified"

                # Optional GPT analysis — graceful fallback
                try:
                    from src.gpt_analyzer import analyze_resume  # noqa: PLC0415
                    gpt_analysis = analyze_resume(text, job_desc)
                    if not str(gpt_analysis).strip():
                        raise ValueError("Empty GPT response")
                except Exception:
                    gpt_analysis = "⚠️ AI analysis unavailable (check API key / quota)."

                final_score = (
                    0.5 * similarity_score
                    + 0.3 * s_score
                    + 0.2 * min(experience / 10, 1.0)
                )
                final_score_pct = round(final_score * 100, 2)

                results.append(
                    {
                        "name": file.name,
                        "score": final_score_pct,
                        "skills": skills,
                        "role": role_display,
                        "ml_roles": ml_roles,
                        "experience": experience,
                        "education": education,
                        "gpt_analysis": gpt_analysis,
                        "decision": get_decision(final_score_pct),
                        "missing_skills": skill_gap(jd_skills, skills),
                    }
                )

            except Exception as e:
                st.error(f"❌ Error processing **{file.name}**: {e}")

            finally:
                progress_bar.progress(
                    (idx + 1) / len(files),
                    text=f"Processed {idx + 1} / {len(files)} resumes…",
                )

    progress_bar.empty()

    # ── Guard: no results ─────────────────────────────────────
    if not results:
        st.error("No resumes could be processed. Please check the uploaded files.")
        st.stop()

    # ── Sort by score ─────────────────────────────────────────
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    df = pd.DataFrame(results)

    # ══════════════════════════════════════════════════════════
    # DASHBOARD — KPI METRICS
    # ══════════════════════════════════════════════════════════
    st.markdown("## 📊 Recruiter Dashboard")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Candidates", len(df))
    m2.metric("Avg Match Score", f"{df['score'].mean():.1f}%")
    m3.metric("Avg Experience", f"{df['experience'].mean():.1f} yrs")
    m4.metric(
        "Top Score",
        f"{df['score'].max():.1f}%",
        delta=f"+{df['score'].max() - df['score'].mean():.1f}% vs avg",
    )

    # ── Charts ────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_bar = px.bar(
            df,
            x="name",
            y="score",
            title="Match Score per Candidate",
            labels={"name": "Resume", "score": "Score (%)"},
            color="score",
            color_continuous_scale=["#ff5252", "#ffb300", "#1de9b6"],
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0bcd8",
            title_font_size=14,
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with chart_col2:
        fig_scatter = px.scatter(
            df,
            x="experience",
            y="score",
            text="name",
            title="Experience vs Match Score",
            labels={"experience": "Experience (yrs)", "score": "Score (%)"},
            color="score",
            color_continuous_scale=["#ff5252", "#ffb300", "#1de9b6"],
            size="score",
            size_max=24,
        )
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0bcd8",
            title_font_size=14,
            showlegend=False,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # RANKED CANDIDATES
    # ══════════════════════════════════════════════════════════
    st.markdown("## 🏆 Ranked Candidates")

    for rank, r in enumerate(results, start=1):

        st.markdown(
            f"<div class='section-header'>#{rank} — {r['name']}</div>",
            unsafe_allow_html=True,
        )

        col_analysis, col_preview, col_insights = st.columns([1.2, 1.8, 1.2])

        # ── Column 1: Full Analysis Card ──────────────────────
        with col_analysis:
            education_str = (
                ", ".join(r["education"]) if r["education"] else "Not detected"
            )
            skills_str = format_skills(r["skills"]) if r["skills"] else "None detected"

            st.markdown(
                f"""
                <div class="card">
                    <h3 style="margin-bottom:0.5rem">{r['name']}</h3>
                    <p style="font-size:1.1rem;margin-bottom:0.75rem">{r['decision']}</p>
                    <p><b>Score:</b> <span class="match-score"
                        style="font-size:1.6rem">{r['score']}%</span></p>
                    <p><b>Experience:</b> {r['experience']} years</p>
                    <p><b>Education:</b> {education_str}</p>
                    <p><b>Matched Role:</b> {r['role']}</p>
                    <p><b>Skills:</b> {skills_str}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(r["score"] / 100)

        # ── Column 2: Resume Preview ──────────────────────────
        with col_preview:
            st.markdown("### 📄 Resume Preview")

            file_obj = raw_texts.get(r["name"])
            if file_obj:
                st.markdown(pdf_iframe(file_obj), unsafe_allow_html=True)
            else:
                st.warning("Preview not available for this resume.")

        # ── Column 3: Insights ────────────────────────────────
        with col_insights:
            st.markdown("### 🎯 Role Confidence")
            if r["ml_roles"]:
                for role_name, role_score in r["ml_roles"]:
                    st.write(f"**{role_name}:** {role_score}%")
                    st.progress(role_score / 100)
            else:
                st.info("No role predictions available.")

            st.markdown("---")

            st.markdown("### 🔍 Skill Gap")
            if r["missing_skills"]:
                for skill in r["missing_skills"]:
                    st.markdown(
                        f"<span class='skill-tag'>❌ {skill}</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.success("✅ No major skill gaps detected!")

            st.markdown("---")

            st.markdown("### 🤖 AI Analysis")
            st.write(r["gpt_analysis"])

        # ── Interview Questions (full width) ──────────────────
        with st.expander(f"🎤 Interview Questions — {r['name']}", expanded=False):
            questions = generate_role_based_questions(r["role"], r["skills"])
            for i, q in enumerate(questions, start=1):
                st.markdown(f"**{i}.** {q}")

        st.markdown("<hr>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # EXPORT / DOWNLOAD
    # ══════════════════════════════════════════════════════════
    st.markdown("## 📥 Export Results")

    export_df = df[["name", "score", "role", "experience", "decision"]].copy()
    export_df["decision"] = export_df["decision"].str.replace(r"<[^>]+>", "", regex=True)

    csv_data = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📊 Download Shortlist (CSV)",
        data=csv_data,
        file_name="shortlisted_candidates.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;opacity:0.5;font-size:0.8rem'>"
    "Advanced AI Resume Screening System — Built with Streamlit"
    "</p>",
    unsafe_allow_html=True,
)