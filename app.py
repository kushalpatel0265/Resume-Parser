import os
import io
import re
import nltk
import PyPDF2
import streamlit as st

# If you haven't downloaded these NLTK data files yet, do so once:
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt_tab")  # Some environments require 'punkt_tab'

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sentence_transformers import SentenceTransformer, util

# Attempt to import Groq LLM (if installed).
try:
    from llama_index.llms.groq import Groq
    from llama_index.core.llms import ChatMessage
    USE_GROQ = True
except ImportError:
    USE_GROQ = False

# Hardcode your Groq API key here or set it in your environment variables
# If you prefer to store it as an environment variable, do:
# export GROQ_API_KEY="YOUR_API_KEY"
# Then in code, you can do:
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY = "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"

########################################
# Helper Functions
########################################

def extract_text_from_pdf(pdf_file) -> str:
    """
    Extracts text from an uploaded PDF file using PyPDF2.
    """
    reader = PyPDF2.PdfReader(pdf_file)
    all_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        all_text.append(text)
    return "\n".join(all_text)

def clean_resume_with_llm(raw_text: str) -> str:
    """
    Uses Groq LLM to clean and standardize the resume text.
    If Groq LLM is unavailable, returns the raw text.
    """
    if not USE_GROQ:
        return raw_text
    system_msg = "You are an AI assistant that cleans and standardizes resume text."
    user_msg = f"Please remove unnecessary formatting and ensure clarity:\n\n{raw_text}"
    llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    messages = [
        ChatMessage(role="system", content=system_msg),
        ChatMessage(role="user", content=user_msg),
    ]
    response = llm.chat(messages)
    return str(response)

def parse_resume_details(cleaned_text: str) -> str:
    """
    Uses Groq LLM to parse the cleaned resume text into structured details (Name, Education, Skills, etc.).
    If Groq LLM is unavailable, returns a fallback message.
    """
    if not USE_GROQ:
        return "Groq LLM not installed or configured. Skipping structured parse."
    system_msg = "You extract key details from resumes (Name, Contact, Education, Experience, Skills)."
    user_msg = f"Parse the following resume and return structured details in bullet points:\n\n{cleaned_text}"
    llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    messages = [
        ChatMessage(role="system", content=system_msg),
        ChatMessage(role="user", content=user_msg),
    ]
    response = llm.chat(messages)
    return str(response)

def lemma_tokenize(text: str) -> set:
    """
    Tokenizes text into words, removes stopwords, lemmatizes tokens,
    and returns a set of lemmas.
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

def lemma_based_match_details(jd_text: str, resume_text: str) -> dict:
    """
    Returns a dict with lemma-based matching details:
      - score: float (0-100)
      - matched: list of lemmas found in both
      - unmatched_jd: list of lemmas in JD but not in resume
      - unmatched_resume: list of lemmas in resume but not in JD
    """
    jd_lemmas = lemma_tokenize(jd_text)
    resume_lemmas = lemma_tokenize(resume_text)
    if not jd_lemmas or not resume_lemmas:
        return {"score": 0.0, "matched": [], "unmatched_jd": [], "unmatched_resume": []}
    intersection = jd_lemmas & resume_lemmas
    union = jd_lemmas | resume_lemmas
    score = (len(intersection) / len(union)) * 100 if union else 0.0
    return {
        "score": round(score, 2),
        "matched": sorted(list(intersection)),
        "unmatched_jd": sorted(list(jd_lemmas - intersection)),
        "unmatched_resume": sorted(list(resume_lemmas - intersection))
    }

def semantic_sentence_matching_details(jd_text: str, resume_text: str, threshold: float = 0.4) -> dict:
    """
    Splits the JD text into sentences, compares each with the entire resume,
    and returns:
      - overall_semantic_score: entire JD vs. resume similarity (0-100)
      - matched_sentences: (sentence, similarity%) if sim >= threshold
      - unmatched_sentences: (sentence, similarity%) if sim < threshold
    """
    # Overall similarity
    model = SentenceTransformer("all-MiniLM-L6-v2")
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    overall_sim = float(util.cos_sim(jd_emb, resume_emb)[0][0])
    overall_score = round(overall_sim * 100, 2)

    # Sentence-level
    sentences = nltk.sent_tokenize(jd_text)
    matched = []
    unmatched = []
    for sent in sentences:
        sent_emb = model.encode(sent, convert_to_tensor=True)
        sim = float(util.cos_sim(sent_emb, resume_emb)[0][0])
        sim_percent = round(sim * 100, 2)
        if sim >= threshold:
            matched.append((sent, sim_percent))
        else:
            unmatched.append((sent, sim_percent))

    return {
        "overall_semantic_score": overall_score,
        "matched_sentences": matched,
        "unmatched_sentences": unmatched
    }

def combined_matching_score(jd_text: str, resume_text: str, lemma_weight: float = 0.5, sem_weight: float = 0.5) -> float:
    """
    Weighted average of lemma-based similarity & entire-text semantic similarity.
    Returns final score (0-100).
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Lemma-based
    lemma_info = lemma_based_match_details(jd_text, resume_text)
    lemma_score = lemma_info["score"]
    
    # Semantic-based (entire text)
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    sim = float(util.cos_sim(jd_emb, resume_emb)[0][0])
    sem_score = round(sim * 100, 2)

    return round((lemma_weight * lemma_score) + (sem_weight * sem_score), 2)


########################################
# Streamlit App
########################################

def main():
    st.title("Resume vs. Job Description Matcher")
    st.write("Upload a PDF resume and paste a job description. The app will use both lemma-based and semantic matching to compute match scores.")

    # 1) Job Description Input
    jd_text = st.text_area("Paste Job Description Here", height=200)

    # 2) PDF Upload
    uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])

    # Action button
    if st.button("Process"):
        if not jd_text.strip():
            st.error("Please paste a job description.")
            return
        if not uploaded_file:
            st.error("Please upload a PDF resume.")
            return

        # Extract the PDF resume
        resume_text_raw = extract_text_from_pdf(uploaded_file)
        st.subheader("Raw Resume Text")
        st.write(resume_text_raw)

        # Clean with LLM (if available)
        st.subheader("Cleaned Resume Text")
        cleaned_resume_text = clean_resume_with_llm(resume_text_raw)
        st.write(cleaned_resume_text)

        # Parse structured details (if LLM available)
        st.subheader("Parsed Resume Details (LLM)")
        details = parse_resume_details(cleaned_resume_text)
        st.write(details)

        # Lemma-based matching info
        lemma_details = lemma_based_match_details(jd_text, cleaned_resume_text)
        st.subheader("Lemma-Based Matching Details")
        st.write(f"**Lemma-Based Similarity Score:** {lemma_details['score']}%")
        st.write("**Matched Lemmas:**")
        st.write(", ".join(lemma_details["matched"]) if lemma_details["matched"] else "No matched lemmas.")
        st.write("**Unmatched JD Lemmas:**")
        st.write(", ".join(lemma_details["unmatched_jd"]) if lemma_details["unmatched_jd"] else "None.")
        st.write("**Unmatched Resume Lemmas:**")
        st.write(", ".join(lemma_details["unmatched_resume"]) if lemma_details["unmatched_resume"] else "None.")

        # Sentence-level semantic
        semantic_info = semantic_sentence_matching_details(jd_text, cleaned_resume_text, threshold=0.4)
        st.subheader("Semantic Matching Details")
        st.write(f"**Overall Semantic Similarity Score:** {semantic_info['overall_semantic_score']}%")
        
        st.write("**Matched JD Sentences (threshold 0.4)**")
        if semantic_info["matched_sentences"]:
            for sent, sim_val in semantic_info["matched_sentences"]:
                st.markdown(f"- {sent} (Score: {sim_val}%)")
        else:
            st.write("No matched sentences.")
        
        st.write("**Unmatched JD Sentences**")
        if semantic_info["unmatched_sentences"]:
            for sent, sim_val in semantic_info["unmatched_sentences"]:
                st.markdown(f"- {sent} (Score: {sim_val}%)")
        else:
            st.write("None.")
        
        # Combined Score
        combined = combined_matching_score(jd_text, cleaned_resume_text, 0.5, 0.5)
        st.subheader(f"Combined Matching Score: {combined}%")


if __name__ == "__main__":
    main()
