import io
import json
import os
import re
from typing import Any, Dict, List

import streamlit as st
from google import genai
from pypdf import PdfReader
from docx import Document


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Resume ATS Analyzer",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Resume ATS Analyzer")
st.caption("Upload a PDF or DOCX resume and get an estimated ATS score, keyword analysis, and AI-powered improvement suggestions.")


# -----------------------------
# Resume extraction
# -----------------------------
def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def extract_docx_text(file_bytes: bytes) -> str:
    document = Document(io.BytesIO(file_bytes))
    parts = [p.text for p in document.paragraphs if p.text.strip()]

    # Include text from tables because many resumes use tables for skills/experience.
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            row_text = " | ".join(c for c in cells if c)
            if row_text:
                parts.append(row_text)

    return "\n".join(parts).strip()


def extract_resume_text(uploaded_file) -> str:
    file_bytes = uploaded_file.getvalue()
    suffix = uploaded_file.name.lower()

    if suffix.endswith(".pdf"):
        return extract_pdf_text(file_bytes)
    if suffix.endswith(".docx"):
        return extract_docx_text(file_bytes)

    raise ValueError("Unsupported file type. Please upload a PDF or DOCX resume.")


# -----------------------------
# Basic ATS checks
# -----------------------------
SECTION_PATTERNS = {
    "contact information": [
        r"\bemail\b", r"\bphone\b", r"\bmobile\b", r"\blinkedin\b", r"\bgithub\b"
    ],
    "summary/objective": [
        r"\bprofessional summary\b", r"\bsummary\b", r"\bobjective\b", r"\bprofile\b"
    ],
    "experience": [
        r"\bwork experience\b", r"\bprofessional experience\b", r"\bexperience\b", r"\bemployment\b"
    ],
    "education": [
        r"\beducation\b", r"\bacademic background\b"
    ],
    "skills": [
        r"\bskills\b", r"\btechnical skills\b", r"\bcore skills\b"
    ],
    "projects": [
        r"\bprojects\b", r"\bacademic projects\b", r"\bpersonal projects\b"
    ],
}


def basic_ats_checks(text: str) -> Dict[str, Any]:
    lower = text.lower()
    words = re.findall(r"\b[\w+#.-]+\b", text)
    word_count = len(words)

    sections = {}
    for section, patterns in SECTION_PATTERNS.items():
        sections[section] = any(re.search(pattern, lower) for pattern in patterns)

    email_ok = bool(re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, re.I))
    phone_ok = bool(re.search(r"(?:\+?\d[\d\s().-]{7,}\d)", text))
    linkedin_ok = "linkedin.com" in lower
    github_ok = "github.com" in lower

    bullet_lines = sum(
        1 for line in text.splitlines()
        if re.match(r"^\s*(?:[-•▪◦*]|\d+[.)])\s+", line)
    )

    action_verbs = {
        "developed", "designed", "built", "created", "implemented", "analyzed",
        "managed", "led", "improved", "optimized", "automated", "tested",
        "deployed", "engineered", "configured", "integrated", "delivered",
        "achieved", "reduced", "increased", "generated", "collaborated"
    }
    found_action_verbs = sorted({
        word.lower().strip(".,;:()[]{}")
        for word in words
        if word.lower().strip(".,;:()[]{}") in action_verbs
    })

    # A simple ATS estimate. This is intentionally presented as an estimate,
    # because real ATS systems use different parsing and ranking rules.
    score = 0
    score += 10 if sections["contact information"] else 0
    score += 15 if sections["summary/objective"] else 0
    score += 20 if sections["experience"] else 0
    score += 15 if sections["education"] else 0
    score += 15 if sections["skills"] else 0
    score += 10 if sections["projects"] else 0
    score += 5 if email_ok else 0
    score += 3 if phone_ok else 0
    score += 2 if (linkedin_ok or github_ok) else 0
    score += 5 if bullet_lines >= 3 else 0

    # Very short extracted text often indicates a scanned/image-only PDF.
    extraction_warning = word_count < 80

    return {
        "estimated_score": min(score, 100),
        "word_count": word_count,
        "sections": sections,
        "email_found": email_ok,
        "phone_found": phone_ok,
        "linkedin_found": linkedin_ok,
        "github_found": github_ok,
        "bullet_count": bullet_lines,
        "action_verbs": found_action_verbs,
        "extraction_warning": extraction_warning,
    }


# -----------------------------
# Gemini analysis
# -----------------------------
def get_gemini_api_key() -> str:
    # Streamlit Cloud: st.secrets["GEMINI_API_KEY"]
    # Local development: environment variable GEMINI_API_KEY
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""

    return key or os.getenv("GEMINI_API_KEY", "")


def get_model_name() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.4-flash")


def analyze_with_gemini(resume_text: str, target_role: str) -> Dict[str, Any]:
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError(
            "Gemini API key is missing. Add GEMINI_API_KEY to Streamlit Secrets "
            "or set it as an environment variable."
        )

    client = genai.Client(api_key=api_key)

    role_instruction = (
        f"The target job role is: {target_role.strip()}."
        if target_role.strip()
        else "No specific target role was provided. Evaluate general ATS readiness."
    )

    prompt = f"""
You are an expert ATS resume reviewer and recruiter.

Analyze the resume below. Give an ESTIMATED ATS score from 0 to 100.
This is not a score from a real ATS vendor; it is an informed estimate based on
machine-readable structure, relevance, keyword alignment, clarity, and measurable impact.

{role_instruction}

Return ONLY valid JSON with this exact structure:
{{
  "ats_score": 0,
  "score_explanation": "short explanation",
  "strengths": ["...", "..."],
  "critical_improvements": [
    {{
      "issue": "...",
      "why_it_matters": "...",
      "recommendation": "..."
    }}
  ],
  "missing_keywords": ["...", "..."],
  "suggested_keywords": ["...", "..."],
  "formatting_warnings": ["...", "..."],
  "rewrites": [
    {{
      "section": "Experience",
      "before": "short quote or description from resume",
      "after": "improved version"
    }}
  ],
  "summary_recommendation": "..."
}}

Rules:
- Do not invent employers, degrees, certifications, projects, dates, metrics, or achievements.
- If a metric is missing, suggest adding a real metric rather than fabricating one.
- Keep missing_keywords relevant to the target role and only recommend terms that genuinely fit the resume.
- Flag tables, columns, graphics, icons, headers/footers, unusual symbols, and excessive formatting only when the extracted text provides evidence or the format commonly creates ATS risk.
- Prioritize concrete improvements.
- If the resume is already strong, still identify realistic refinements.
- Keep the JSON concise enough to be practical.

RESUME:
----------------
{resume_text[:50000]}
----------------
"""

    response = client.models.generate_content(
        model=get_model_name(),
        contents=prompt,
        config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    )

    raw = (response.text or "").strip()

    # Defensive cleanup in case a model returns a fenced JSON block.
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Gemini returned an invalid JSON response. Please try again. Details: {exc}"
        ) from exc

    if not isinstance(result, dict):
        raise RuntimeError("Gemini returned an unexpected response format.")

    return result


# -----------------------------
# UI helpers
# -----------------------------
def show_score(score: int) -> None:
    score = max(0, min(100, int(score)))

    if score >= 80:
        label = "Excellent ATS readiness"
    elif score >= 65:
        label = "Good, but needs improvement"
    elif score >= 50:
        label = "Moderate ATS readiness"
    else:
        label = "Needs significant improvement"

    st.metric("Estimated ATS Score", f"{score}/100")
    st.progress(score / 100)
    st.caption(label)


def safe_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


# -----------------------------
# Main app
# -----------------------------
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    target_role = st.text_input(
        "Target job title (optional)",
        placeholder="e.g. Data Scientist",
        help="Adding a target role makes keyword and relevance analysis more useful.",
    )

    st.info(
        "Privacy tip: resumes can contain personal information. "
        "Avoid storing API keys in your code or GitHub repository."
    )

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx"],
    help="PDF and DOCX are supported. Text-based PDFs work best.",
)

if uploaded_file:
    st.write(f"**File:** {uploaded_file.name}")

    try:
        resume_text = extract_resume_text(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the resume: {exc}")
        st.stop()

    if not resume_text:
        st.error("No readable text was found. If this is a scanned/image-only PDF, please upload a text-based PDF or DOCX.")
        st.stop()

    basic = basic_ats_checks(resume_text)

    if basic["extraction_warning"]:
        st.warning(
            "Only a small amount of text was extracted. Your resume may be scanned/image-based, "
            "or the PDF may have an unusual layout. ATS results may be less reliable."
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Extracted words", basic["word_count"])
    with col2:
        st.metric("Bullet points", basic["bullet_count"])
    with col3:
        st.metric("Basic ATS estimate", f'{basic["estimated_score"]}/100')

    if st.button("🔍 Analyze Resume with Gemini", type="primary", use_container_width=True):
        with st.spinner("Analyzing your resume..."):
            try:
                ai_result = analyze_with_gemini(resume_text, target_role)
            except Exception as exc:
                st.error(str(exc))
                st.stop()

        ai_score = ai_result.get("ats_score", basic["estimated_score"])
        try:
            ai_score = int(float(ai_score))
        except (TypeError, ValueError):
            ai_score = basic["estimated_score"]

        st.divider()
        st.subheader("📊 ATS Score")
        show_score(ai_score)

        explanation = ai_result.get("score_explanation", "")
        if explanation:
            st.write(explanation)

        st.subheader("✅ Strengths")
        strengths = safe_list(ai_result.get("strengths"))
        if strengths:
            for item in strengths:
                st.markdown(f"- {item}")
        else:
            st.write("No strengths were returned.")

        st.subheader("🚨 Critical Improvements")
        improvements = ai_result.get("critical_improvements", [])
        if isinstance(improvements, list) and improvements:
            for item in improvements:
                if isinstance(item, dict):
                    issue = item.get("issue", "Improvement")
                    why = item.get("why_it_matters", "")
                    recommendation = item.get("recommendation", "")
                    with st.expander(str(issue)):
                        if why:
                            st.write(f"**Why it matters:** {why}")
                        if recommendation:
                            st.write(f"**Recommendation:** {recommendation}")
                else:
                    st.markdown(f"- {item}")
        else:
            st.write("No critical improvements were returned.")

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("🔑 Missing Keywords")
            keywords = safe_list(ai_result.get("missing_keywords"))
            if keywords:
                st.write(", ".join(keywords))
            else:
                st.write("No major missing keywords identified.")

        with col_b:
            st.subheader("💡 Suggested Keywords")
            keywords = safe_list(ai_result.get("suggested_keywords"))
            if keywords:
                st.write(", ".join(keywords))
            else:
                st.write("No additional keywords suggested.")

        st.subheader("🧹 Formatting Warnings")
        warnings = safe_list(ai_result.get("formatting_warnings"))
        if warnings:
            for item in warnings:
                st.markdown(f"- {item}")
        else:
            st.success("No major formatting warnings were identified from the available resume text.")

        st.subheader("✍️ Suggested Rewrites")
        rewrites = ai_result.get("rewrites", [])
        if isinstance(rewrites, list) and rewrites:
            for item in rewrites:
                if isinstance(item, dict):
                    section = item.get("section", "Resume section")
                    before = item.get("before", "")
                    after = item.get("after", "")
                    with st.expander(str(section)):
                        if before:
                            st.markdown(f"**Before:** {before}")
                        if after:
                            st.markdown(f"**After:** {after}")
                else:
                    st.markdown(f"- {item}")
        else:
            st.write("No specific rewrites were returned.")

        summary = ai_result.get("summary_recommendation", "")
        if summary:
            st.subheader("🎯 Overall Recommendation")
            st.info(summary)

        with st.expander("🔎 Basic parser checks"):
            for name, value in basic["sections"].items():
                st.write(f"{'✅' if value else '❌'} {name.title()}")

            st.write(f"Email found: {'Yes' if basic['email_found'] else 'No'}")
            st.write(f"Phone found: {'Yes' if basic['phone_found'] else 'No'}")
            st.write(f"LinkedIn found: {'Yes' if basic['linkedin_found'] else 'No'}")
            st.write(f"GitHub found: {'Yes' if basic['github_found'] else 'No'}")

            if basic["action_verbs"]:
                st.write("Action verbs detected:", ", ".join(basic["action_verbs"]))

else:
    st.markdown(
        """
### How it works
1. Upload a **PDF or DOCX** resume.
2. Optionally enter the **target job title**.
3. The app extracts the resume text.
4. Gemini Flash evaluates ATS readiness, keywords, structure, and content.
5. You receive an **estimated ATS score + specific improvements + rewrite suggestions**.

> **Important:** ATS scores are estimates. Different applicant-tracking systems use different parsing and ranking rules, so no third-party app can guarantee an exact score for every ATS.
"""
    )
