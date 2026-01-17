import streamlit as st
import joblib
import re
import ftfy
import nltk
import pdfplumber    # for extracting text from pdf file
from nltk.corpus import stopwords

# ===============================
# Load Models
# ===============================
model = joblib.load("Knn_model.pkl")
encoder = joblib.load("label_encoder.pkl")
tfidf = joblib.load("tfidf.pkl")

# ===============================
# NLTK Setup
# ===============================
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# ===============================
# Text Cleaning Function
# ===============================
def clean_text(text):
    text = ftfy.fix_text(text)
    text = text.lower()

    text = re.sub(r"[•●▪■►✓❖]", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"#\S+|@\S+", " ", text)

    text = re.sub(
        r"[%s]" % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),
        " ",
        text,
    )

    # keep tech tokens like python3, html5
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# ===============================
# PDF Text Extraction
# ===============================
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ===============================
# Skills Section Extraction
# ===============================
def extract_skills_section(text):
    text = text.lower()
    lines = text.split("\n")

    skills_lines = []
    capture = False

    for line in lines:
        line = line.strip()

        # Start capturing after 'skills' line
        if "skills" in line:
            capture = True
            continue

        # Stop capturing when a new section starts
        if capture and any(word in line for word in ["experience", "education", "projects", "summary"]):
            break

        # Collect lines in skills section
        if capture:
            skills_lines.append(line)

    # Join lines to return single text block
    return " ".join(skills_lines)

# ===============================
# Individual Skill Extraction
# ===============================
SKILLS_LIST = [
    # Programming
    "python", "java", "c", "c++", "c#", "javascript", "typescript",
    # Web
    "html", "css", "bootstrap", "tailwind", "react", "angular", "vue",
    "django", "flask", "fastapi", "nodejs", "express",
    # Data / ML
    "numpy", "pandas", "matplotlib", "seaborn", "scikit-learn",
    "tensorflow", "pytorch", "keras", "machine learning", "deep learning", "nlp",
    # Database
    "mysql", "postgresql", "mongodb", "sqlite", "redis",
    # DevOps / Cloud
    "aws", "azure", "gcp", "docker", "kubernetes",
    # Tools
    "git", "github", "linux", "postman",
    # Others
    "blockchain", "solidity", "ethereum"
]

def extract_individual_skills(text):
    found = set()
    text = text.lower()
    for skill in SKILLS_LIST:
        if re.search(rf"\b{re.escape(skill)}\b", text):
            found.add(skill)
    return sorted(found)

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Resume Skill Extractor", layout="centered")
st.title("📄 Resume Skill Extraction & Screening")

uploaded_resume = st.file_uploader(
    "Upload Resume (PDF or TXT)", type=["pdf", "txt"]
)

if uploaded_resume:
    # ---------- Read File ----------
    if uploaded_resume.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_resume)
    else:
        resume_text = uploaded_resume.read().decode("utf-8", errors="ignore")

    # ---------- Extract Skills Section ----------
    skills_section = extract_skills_section(resume_text)

    if not skills_section.strip():
        st.warning("⚠️ Skills section not detected. Using full resume.")
        skills_section = resume_text

    # ---------- Clean Skills Text ----------
    cleaned_skills_text = clean_text(skills_section)

    # ---------- Extract Individual Skills ----------
    skills_found = extract_individual_skills(skills_section)

    st.subheader("🛠️ Extracted Skills")
    if skills_found:
        st.write(", ".join(skills_found))
    else:
        st.info("No predefined skills detected.")

    # ---------- Optional Prediction ----------
    if st.button("🔍 Predict Job Category (Using Skills Only)"):
        vectorized_text = tfidf.transform([cleaned_skills_text])
        prediction = model.predict(vectorized_text)
        predicted_category = encoder.inverse_transform(prediction)

        st.success(f"Predicted Category: **{predicted_category[0]}**")
