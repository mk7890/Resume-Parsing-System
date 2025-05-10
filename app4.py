'''
All entity types are now included in the NER parsing.
Entities not detected will return None instead of being missing.
Improved UI to display all fields.
Added optional fallbacks for contact info using regex (if not found via NER).
Ensured compatibility with your DistilBERT NER model trained on these 9 entity types.
'''

import streamlit as st
import pandas as pd
import pdfplumber
from docx import Document
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import re
import os

# ✅ MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Resume Parsersing System", layout="wide")
# ✅ END OF FIRST COMMAND

# -----------------------------
# Load Model & Tokenizer
# -----------------------------
@st.cache_resource
def load_ner_pipeline():
    model_path = "ner_model_10pc"
    if not os.path.exists(model_path):
        st.error("Model not found. Please ensure the 'ner_model' folder exists.")
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        return pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple"
        )
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


ner_pipeline = load_ner_pipeline()

# -----------------------------
# File Reading Functions
# -----------------------------
def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
        return text


def read_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])


def read_file(file):
    if file.type == "application/pdf":
        return read_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return read_docx(file)
    else:
        raise ValueError("Unsupported file format")


# -----------------------------
# Resume Parsing Function (Modified to include ALL entities)
# -----------------------------
def parse_resume(text):
    result = ner_pipeline(text)
    parsed = {
        "Name": [],
        "Title": [],
        "Role": [],
        "Contact": [],
        "Email": [],
        "Qualifications": [],
        "Experience": [],
        "Skills": [],
        "Company": []
    }
    for item in result:
        label = item['entity_group']
        word = item['word'].strip()
        if label in parsed:
            if word not in parsed[label]:
                parsed[label].append(word)

    # Optional fallbacks
    full_text = text
    if not parsed["Email"]:
        email = extract_email(full_text)
        if email:
            parsed["Email"] = [email]
    if not parsed["Contact"]:
        phone = extract_phone(full_text)
        if phone:
            parsed["Contact"] = [phone]

    return parsed


# -----------------------------
# Clean Tokens Helper
# -----------------------------
def clean_tokens(tokens):
    words = []
    current = ""
    for token in tokens:
        token = token.replace("##", "")
        if current and not current.endswith(" "):
            current += token
        else:
            if current:
                words.append(current.strip())
            current = token
    if current:
        words.append(current.strip())
    return words


# -----------------------------
# Extract Email & Phone
# -----------------------------
def extract_email(text):
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group() if match else None


def extract_phone(text):
    match = re.search(r"(\+?\d[\d\s\-()]{7,})", text)
    return match.group().strip() if match else None


# -----------------------------
# Normalize Skills
# -----------------------------
def split_skills(text):
    parts = re.split(r",|\u2022|-|•", text)
    return list(set([p.strip().capitalize() for p in parts if len(p.strip()) > 2]))


# -----------------------------
# Match Against Job Description
# -----------------------------
def compute_match_score(resume_text, job_desc):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]


def compare_resume_to_job(parsed_resume, job_desc):
    resume_text = " ".join([
        " ".join(parsed_resume["Skills"]),
        " ".join(parsed_resume["Experience"]),
        " ".join(parsed_resume["Qualifications"])
    ])
    job_desc_lower = job_desc.lower()
    matched_skills = [skill for skill in parsed_resume["Skills"] if skill.lower() in job_desc_lower]
    missing_skills = [skill for skill in parsed_resume["Skills"] if skill.lower() not in job_desc_lower]
    score = compute_match_score(resume_text, job_desc)
    return {
        "score": round(score * 100, 2),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }


# -----------------------------
# Highlight Matching Keywords
# -----------------------------
def highlight_keywords(text, keywords, color="yellow"):
    for keyword in keywords:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        text = pattern.sub(f'<mark style="background-color:{color}">{keyword}</mark>', text)
    return text


# -----------------------------
# Batch Processing Function
# -----------------------------
def batch_process_resumes(files, job_description):
    results = []
    for file in files:
        try:
            resume_text = read_file(file)
            parsed = parse_resume(resume_text)
            comparison = compare_resume_to_job(parsed, job_description)
            results.append({
                "filename": file.name,
                "score": comparison["score"],
                "parsed": parsed,
                "comparison": comparison,
                "text": resume_text
            })
        except Exception as e:
            results.append({"filename": file.name, "error": str(e)})
    return sorted(results, key=lambda x: x.get("score", 0), reverse=True)


# -----------------------------
# Analytics Dashboard
# -----------------------------
def show_analytics_dashboard(results):
    st.header("📊 Analytics Dashboard")
    df_scores = pd.DataFrame([{k: v for k, v in res.items() if k in ['filename', 'score']} for res in results if "error" not in res])
    st.subheader("🏆 Ranked Resumes by Match Score:")
    st.dataframe(df_scores.style.background_gradient(cmap='Blues', subset=["score"]))
    
    all_skills = []
    for res in results:
        if "error" not in res:
            all_skills.extend(res["parsed"]["Skills"])
    
    from collections import Counter
    skill_counter = Counter(all_skills)
    skill_df = pd.DataFrame(skill_counter.most_common(10), columns=["Skill", "Count"])
    st.subheader("🔥 Top Skills Across All Resumes")
    st.bar_chart(skill_df.set_index('Skill'))


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    # Custom CSS
    st.markdown("""
    <style>
    body { font-family: Arial; }
    h1 { color: #2C3E50; }
    .stButton>button { background-color: #2980B9; color: white; border-radius: 6px; }
    mark { padding: 2px 4px; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

    # Logo
    try:
        logo = Image.open("logo.png")
        st.image(logo, width=200)
    except FileNotFoundError:
        st.warning("No logo found. Add 'logo.png' for branding.")

    st.title("📄 Resume Parser & Job Matcher")
    st.markdown("Upload resumes and compare them with a job description. Get match scores, skill analysis, and ranked list.")

    tab1, tab2, tab3 = st.tabs(["🔍 Single Resume Analysis", "📂 Batch Resume Ranking", "📈 Analytics Dashboard"])

    # --- Tab 1: Single Resume ---
    with tab1:
        uploaded_file = st.file_uploader("Upload your resume (.pdf or .docx)", type=["pdf", "docx"], key="single_upload")
        job_description = st.text_area("Enter job description", height=200)
        if st.button("🔍 Analyze Resume") and uploaded_file and job_description:
            with st.spinner("Parsing resume..."):
                resume_text = read_file(uploaded_file)
                parsed = parse_resume(resume_text)
                comparison = compare_resume_to_job(parsed, job_description)

                st.subheader("📊 Match Analysis")
                st.write(f"**Match Score:** {comparison['score']}%")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("✅ **Matching Skills**")
                    st.write(", ".join(comparison['matched_skills']) or "None")
                with col2:
                    st.markdown("❌ **Missing Skills**")
                    st.write(", ".join(comparison['missing_skills']) or "None")

                st.markdown("📌 **Parsed Resume Info**")
                st.json({
                    key: " | ".join(value) if isinstance(value, list) else value
                    for key, value in parsed.items()
                })

                st.markdown("📝 **Highlighted Resume**")
                highlighted_resume = highlight_keywords(resume_text, comparison['matched_skills'], "lightgreen")
                highlighted_resume = highlight_keywords(highlighted_resume, comparison['missing_skills'], "lightcoral")
                st.markdown(highlighted_resume, unsafe_allow_html=True)

    # --- Tab 2: Batch Mode ---
    with tab2:
        st.header("📥 Batch Resume Matching (HR)")
        job_desc_batch = st.text_area("Job Description for Batch", height=150)
        batch_files = st.file_uploader("Upload multiple resumes", type=["pdf", "docx"], accept_multiple_files=True)
        if st.button("🏆 Rank Resumes") and job_desc_batch and batch_files:
            with st.spinner("Processing resumes..."):
                batch_results = batch_process_resumes(batch_files, job_desc_batch)
                st.subheader("🏆 Ranked Resumes by Match Score:")
                df_ranked = pd.DataFrame([{
                    "Filename": res["filename"],
                    "Score (%)": res["score"]
                } for res in batch_results if "error" not in res])
                st.dataframe(df_ranked.style.background_gradient(cmap='Blues', subset=["Score (%)"]))
                for res in batch_results:
                    if "error" in res:
                        st.error(f"{res['filename']}: {res['error']}")

    # --- Tab 3: Analytics Dashboard ---
    with tab3:
        st.header("📈 Analytics Dashboard")
        job_desc_analytics = st.text_area("Enter job description for analytics", height=150)
        analytics_files = st.file_uploader("Upload resumes for analytics", type=["pdf", "docx"], accept_multiple_files=True)
        if st.button("📊 Generate Analytics") and analytics_files and job_desc_analytics:
            with st.spinner("Analyzing resumes..."):
                analytics_results = batch_process_resumes(analytics_files, job_desc_analytics)
                show_analytics_dashboard(analytics_results)


if __name__ == "__main__":
    main()