import streamlit as st
import PyPDF2
import re
import io
import nltk

# NLTK setup
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt_tab")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Attempt to import Groq LLM
try:
    from llama_index.llms.groq import Groq
    from llama_index.core.llms import ChatMessage
    USE_GROQ = True
except ImportError:
    USE_GROQ = False

# Attempt to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer, util
    USE_SEMANTIC = True
except ImportError:
    USE_SEMANTIC = False

# Hardcode your Groq API key (if you have one)
GROQ_API_KEY = "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"

# Known skill keywords for partial parse
KNOWN_SKILLS = ["python", "java", "aws", "machine learning", "html", "css", "javascript"]

########################################
# Helper Functions
########################################

def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract text from an uploaded PDF using PyPDF2.
    """
    reader = PyPDF2.PdfReader(pdf_file)
    all_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        all_text.append(text)
    return "\n".join(all_text)

def code_based_parse(resume_text: str, known_skills=None) -> dict:
    """
    1. Phone (regex)
    2. Email (regex)
    3. Skills detection from known list
    Return partial parse dict
    """
    # Regex for phone
    phone_pattern = r'\+?\d[\d\s\-.()]{7,}\d'
    phones = re.findall(phone_pattern, resume_text)

    # Regex for email
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    emails = re.findall(email_pattern, resume_text)

    # Skills
    detected_skills = []
    if known_skills:
        lower_txt = resume_text.lower()
        for skill in known_skills:
            if skill.lower() in lower_txt:
                detected_skills.append(skill)

    return {
        "phones": list(set(phones)),
        "emails": list(set(emails)),
        "skills": list(set(detected_skills))
    }

def finalize_summary_with_llm(resume_text: str, partial_data: dict) -> str:
    """
    LLM verifies partial parse, removing incorrect info.
    Returns bullet-point subpoints (Education, Experience, etc.).
    """
    if not USE_GROQ:
        return f"(No Groq LLM installed)\nPartial parse:\n{partial_data}"

    system_msg = "You finalize resume data, removing incorrect info, returning subpoints like Education, Experience, Skills, etc."
    user_msg = f"""
Here is a partial parse from code logic:
Phones: {partial_data.get("phones", [])}
Emails: {partial_data.get("emails", [])}
Skills: {partial_data.get("skills", [])}

Resume text:
{resume_text}

INSTRUCTIONS:
1) Verify these fields. If any are incorrect, remove or fix them.
2) Summarize the entire resume into bullet points with subheadings (Education, Experience, Projects, Skills, etc.).
3) Return the final summary in bullet points.
"""
    try:
        llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
        messages = [
            ChatMessage(role="system", content=system_msg),
            ChatMessage(role="user", content=user_msg),
        ]
        response = llm.chat(messages)
        return str(response)
    except Exception as e:
        return f"LLM call failed: {e}\nPartial parse:\n{partial_data}"

#########################
# Matching Approaches
#########################

def lemma_tokenize(text: str) -> set:
    """
    Tokenize, remove stopwords, lemmatize => set of lemmas.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    sents = nltk.sent_tokenize(text)
    lemmas = set()
    for s in sents:
        tokens = nltk.word_tokenize(s)
        for t in tokens:
            tl = t.lower()
            if tl.isalnum() and tl not in stop_words:
                lemmas.add(lemmatizer.lemmatize(tl))
    return lemmas

def lemma_based_match(jd_text: str, resume_text: str) -> dict:
    """
    Jaccard similarity => 0..100
    Return matched/unmatched tokens
    """
    jd_lemmas = lemma_tokenize(jd_text)
    res_lemmas = lemma_tokenize(resume_text)

    if not jd_lemmas or not res_lemmas:
        return {"score": 0.0, "matched": [], "jd_only": [], "resume_only": []}

    inter = jd_lemmas & res_lemmas
    union = jd_lemmas | res_lemmas
    score = (len(inter) / len(union)) * 100
    return {
        "score": round(score, 2),
        "matched": sorted(list(inter)),
        "jd_only": sorted(list(jd_lemmas - inter)),
        "resume_only": sorted(list(res_lemmas - inter))
    }

def semantic_match(jd_text: str, resume_text: str, threshold=0.4) -> dict:
    """
    Overall text similarity + sentence-level JD lines
    """
    if not USE_SEMANTIC:
        return {
            "overall_score": 0.0,
            "matched_sents": [],
            "unmatched_sents": [],
            "note": "No sentence-transformers installed"
        }
    model = SentenceTransformer("all-MiniLM-L6-v2")
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    res_emb = model.encode(resume_text, convert_to_tensor=True)
    sim_val = float(util.cos_sim(jd_emb, res_emb)[0][0])
    overall_score = round(sim_val * 100, 2)

    matched_sents = []
    unmatched_sents = []
    lines = nltk.sent_tokenize(jd_text)
    for line in lines:
        line_emb = model.encode(line, convert_to_tensor=True)
        line_sim = float(util.cos_sim(line_emb, res_emb)[0][0])
        pct = round(line_sim * 100, 2)
        if line_sim >= threshold:
            matched_sents.append((line, pct))
        else:
            unmatched_sents.append((line, pct))

    return {
        "overall_score": overall_score,
        "matched_sents": matched_sents,
        "unmatched_sents": unmatched_sents
    }

def combined_score(lemma_val: float, sem_val: float, lemma_weight=0.5, sem_weight=0.5) -> float:
    """
    Weighted average => final 0..100
    """
    return round(lemma_val * lemma_weight + sem_val * sem_weight, 2)

########################################
# Streamlit App
########################################

def main():
    st.title("Resume + LLM Summarization & Job Matching")
    st.write("""
    1) Upload a PDF resume.
    2) We'll do partial code-based parse for phone/email/skills.
    3) A Groq LLM can refine & remove incorrect info, returning a final bullet summary.
    4) Provide a job description, and we'll do lemma-based vs. semantic matches.
    """)

    # 1) Upload PDF
    uploaded_file = st.file_uploader("Upload your resume PDF", type=["pdf"])
    # 2) Job Description
    jd_text = st.text_area("Paste your Job Description here", height=150)

    if st.button("Process"):
        if not uploaded_file:
            st.error("No PDF file uploaded.")
            return
        if not jd_text.strip():
            st.error("No job description provided.")
            return

        # Extract text
        pdf_stream = io.BytesIO(uploaded_file.read())
        resume_raw = extract_text_from_pdf(pdf_stream)
        st.subheader("Raw Resume Text")
        st.text_area("Raw Resume Text", resume_raw, height=200)

        # Code-based partial parse
        partial_data = code_based_parse(resume_raw, known_skills=KNOWN_SKILLS)
        st.subheader("Partial Parse (Code-based)")
        st.json(partial_data)

        # LLM finalization
        final_summary = finalize_summary_with_llm(resume_raw, partial_data)
        st.subheader("Finalized Summary (LLM)")
        st.write(final_summary)

        # Lemma-based matching
        lemma_result = lemma_based_match(jd_text, resume_raw)
        lemma_score = lemma_result["score"]
        st.subheader("Lemma-Based Matching")
        st.write(f"Score: {lemma_score}%")
        st.write("Matched tokens:", lemma_result["matched"])
        st.write("JD-only tokens:", lemma_result["jd_only"])
        st.write("Resume-only tokens:", lemma_result["resume_only"])

        # Semantic matching
        sem_result = semantic_match(jd_text, resume_raw, threshold=0.4)
        sem_score = sem_result["overall_score"]
        st.subheader("Semantic Matching")
        st.write(f"Overall Score: {sem_score}%")
        st.write("Matched JD Sentences (threshold=0.4):")
        if sem_result["matched_sents"]:
            for line, pct in sem_result["matched_sents"]:
                st.markdown(f"- {line} ({pct}%)")
        else:
            st.write("No matched lines.")
        st.write("Unmatched JD Sentences:")
        if sem_result["unmatched_sents"]:
            for line, pct in sem_result["unmatched_sents"]:
                st.markdown(f"- {line} ({pct}%)")
        else:
            st.write("None.")

        # Combined Score
        combo = combined_score(lemma_score, sem_score, 0.5, 0.5)
        st.subheader(f"Combined Matching Score: {combo}%")

if __name__ == "__main__":
    main()
