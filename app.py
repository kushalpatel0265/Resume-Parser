import re
import io
import nltk
import PyPDF2
import streamlit as st

# Download NLTK data (only needed the first time)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt_tab")  # Some environments need this

########################
# Attempt to import Groq LLM
########################
try:
    from llama_index.llms.groq import Groq
    from llama_index.core.llms import ChatMessage
    USE_GROQ = True
except ImportError:
    USE_GROQ = False

########################
# Attempt to import sentence-transformers
########################
try:
    from sentence_transformers import SentenceTransformer, util
    USE_SEMANTIC = True
except ImportError:
    USE_SEMANTIC = False

########################
# NLTK Tools for partial parse & lemma approach
########################
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

########################
# Hardcoded API key
########################
GROQ_API_KEY = "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"  # Replace if you have a valid key

########################
# Helper Functions
########################

def extract_text_from_pdf(pdf_file) -> str:
    """Extracts text from a PDF file using PyPDF2."""
    reader = PyPDF2.PdfReader(pdf_file)
    all_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        all_text.append(text)
    return "\n".join(all_text)


def code_based_extraction(resume_text: str, known_skills=None) -> dict:
    """
    Partial parse: phone, email, known skill detection.
    Returns a dict of partial_data for LLM refinement.
    """
    # Regex for phone
    phone_pattern = r'\+?\d[\d\s\-.()]{7,}\d'
    phones = re.findall(phone_pattern, resume_text)

    # Regex for email
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    emails = re.findall(email_pattern, resume_text)

    # Basic skill detection
    detected_skills = []
    if known_skills:
        text_lower = resume_text.lower()
        for skill in known_skills:
            if skill.lower() in text_lower:
                detected_skills.append(skill)

    return {
        "phones": list(set(phones)),
        "emails": list(set(emails)),
        "skills": list(set(detected_skills))
    }


def refine_with_llm(resume_text: str, partial_data: dict) -> str:
    """
    Calls Groq LLM (if available) to confirm/correct partial_data.
    If missing or invalid, returns a fallback message.
    """
    if not USE_GROQ:
        return "Groq LLM not installed. Partial data:\n" + str(partial_data)

    # Build a user prompt
    system_msg = "You are an AI assistant that refines partial resume data."
    user_msg = f"""
We extracted by code logic:
- Phones: {partial_data.get("phones", [])}
- Emails: {partial_data.get("emails", [])}
- Skills: {partial_data.get("skills", [])}

Resume text:
{resume_text}

1) Confirm or correct these fields.
2) Provide other relevant info (Name, Education, Experience, etc.).
3) Return final data in bullet points or JSON.
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
        return f"LLM call failed: {e}\n\nPartial data: {partial_data}"


########################
# LEMMA-BASED MATCHING
########################

def lemma_tokenize(text: str) -> set:
    """
    Tokenize text into words, remove stopwords, lemmatize them, return set of lemmas.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    sentences = nltk.sent_tokenize(text)
    lemma_set = set()
    for sent in sentences:
        tokens = nltk.word_tokenize(sent)
        for token in tokens:
            token_lower = token.lower()
            if token_lower.isalnum() and token_lower not in stop_words:
                lemma_set.add(lemmatizer.lemmatize(token_lower))
    return lemma_set

def lemma_based_similarity(jd_text: str, resume_text: str) -> dict:
    """
    Return a dict with Jaccard similarity (0..100) plus matched/unmatched tokens.
    """
    jd_lemmas = lemma_tokenize(jd_text)
    resume_lemmas = lemma_tokenize(resume_text)

    if not jd_lemmas or not resume_lemmas:
        return {"score": 0.0, "matched": [], "unmatched_jd": [], "unmatched_resume": []}

    intersection = jd_lemmas & resume_lemmas
    union = jd_lemmas | resume_lemmas
    jaccard = (len(intersection) / len(union)) * 100
    return {
        "score": round(jaccard, 2),
        "matched": sorted(list(intersection)),
        "unmatched_jd": sorted(list(jd_lemmas - intersection)),
        "unmatched_resume": sorted(list(resume_lemmas - intersection))
    }


########################
# SEMANTIC MATCHING
########################

def semantic_sentence_matching_details(jd_text: str, resume_text: str, threshold: float = 0.4) -> dict:
    """
    Splits JD into sentences, compares each to entire resume embedding,
    returns overall similarity + matched/unmatched lines.
    """
    if not USE_SEMANTIC:
        return {
            "overall_score": 0.0,
            "matched_sentences": [],
            "unmatched_sentences": [],
            "note": "semantic not installed"
        }

    model = SentenceTransformer("all-MiniLM-L6-v2")
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    res_emb = model.encode(resume_text, convert_to_tensor=True)

    # Overall
    sim = float(util.cos_sim(jd_emb, res_emb)[0][0])
    overall_score = round(sim * 100, 2)

    sentences = nltk.sent_tokenize(jd_text)
    matched = []
    unmatched = []
    for sent in sentences:
        s_emb = model.encode(sent, convert_to_tensor=True)
        s_sim = float(util.cos_sim(s_emb, res_emb)[0][0])
        s_pct = round(s_sim * 100, 2)
        if s_sim >= threshold:
            matched.append((sent, s_pct))
        else:
            unmatched.append((sent, s_pct))

    return {
        "overall_score": overall_score,
        "matched_sentences": matched,
        "unmatched_sentences": unmatched
    }

def combined_score(lemma_score: float, sem_score: float, lemma_weight=0.5, sem_weight=0.5) -> float:
    """
    Weighted average of lemma-based and semantic-based approach. Return 0..100
    """
    return round((lemma_weight * lemma_score) + (sem_weight * sem_score), 2)


########################
# STREAMLIT APP
########################

def main():
    st.title("Resume vs. Job Description Matcher")
    st.write("Paste a Job Description and upload a PDF resume for partial code-based + LLM-based parsing and matching.")
    
    # 1) Job Description
    jd_text = st.text_area("Job Description:", height=200)

    # 2) PDF Upload
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    if st.button("Process"):
        if not jd_text.strip():
            st.error("Please provide a Job Description.")
            return
        if not uploaded_file:
            st.error("Please upload a PDF resume.")
            return

        # Extract raw text
        resume_text = extract_text_from_pdf(uploaded_file)
        st.subheader("Raw Resume Text")
        st.text_area("Raw Resume", resume_text, height=200)

        # Code-based partial extraction
        known_skills = ["python", "java", "aws", "machine learning", "html", "css", "javascript"]
        partial_data = code_based_extraction(resume_text, known_skills=known_skills)
        st.subheader("Partial Data (Code-Based)")
        st.write(partial_data)

        # LLM refinement
        refined_llm_output = refine_with_llm(resume_text, partial_data)
        st.subheader("LLM-Refined Output")
        st.write(refined_llm_output)

        # Lemma-based matching
        lemma_result = lemma_based_similarity(jd_text, resume_text)
        lemma_score = lemma_result["score"]

        st.subheader("Lemma-Based Matching")
        st.write(f"**Score:** {lemma_score}%")
        st.write(f"**Matched Lemmas:** {lemma_result['matched']}")
        st.write(f"**Unmatched JD Lemmas:** {lemma_result['unmatched_jd']}")
        st.write(f"**Unmatched Resume Lemmas:** {lemma_result['unmatched_resume']}")

        # Semantic matching
        sem_details = semantic_sentence_matching_details(jd_text, resume_text, threshold=0.4)
        sem_score = sem_details["overall_score"]

        st.subheader("Semantic Matching (Line-by-Line JD)")
        st.write(f"**Overall Score:** {sem_score}%")

        st.write("**Matched JD Sentences:**")
        for line, pct in sem_details["matched_sentences"]:
            st.markdown(f"- {line} (Score: {pct}%)")
        st.write("**Unmatched JD Sentences:**")
        for line, pct in sem_details["unmatched_sentences"]:
            st.markdown(f"- {line} (Score: {pct}%)")

        # Combined Score
        combo = combined_score(lemma_score, sem_score, lemma_weight=0.5, sem_weight=0.5)
        st.subheader(f"Combined Final Score: {combo}%")

if __name__ == "__main__":
    main()
