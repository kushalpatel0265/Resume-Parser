import os
import io
import re
import nltk
import PyPDF2
import streamlit as st

############################
# NLTK Data Setup
############################
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt_tab")  # Some environments require this

############################
# Attempt to Import Groq LLM
############################
try:
    from llama_index.llms.groq import Groq
    from llama_index.core.llms import ChatMessage
    USE_GROQ = True
except ImportError:
    USE_GROQ = False

############################
# Attempt to Import Sentence Transformers
############################
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError as e:
    # If this fails, the app will show an error in the Streamlit interface.
    st.error("Could not import sentence_transformers. Please ensure it's installed correctly.")
    raise e

############################
# Additional NLTK Tools
############################
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

############################
# Hardcoded Groq API Key
############################
GROQ_API_KEY = "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"

############################
# Helper Functions
############################

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
    Uses Groq LLM to parse the cleaned resume text into structured details
    (e.g., Name, Contact, Education, Experience, Skills).
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
    Computes lemma-based matching details using Jaccard similarity.
    Returns:
      - score: Jaccard similarity percentage (0-100)
      - matched: list of common lemmas
      - unmatched_jd: lemmas in JD but not in resume
      - unmatched_resume: lemmas in resume but not in JD
    """
    jd_lemmas = lemma_tokenize(jd_text)
    resume_lemmas = lemma_tokenize(resume_text)

    if not jd_lemmas or not resume_lemmas:
        return {
            "score": 0.0,
            "matched": [],
            "unmatched_jd": [],
            "unmatched_resume": []
        }

    intersection = jd_lemmas & resume_lemmas
    union = jd_lemmas | resume_lemmas
    jaccard = (len(intersection) / len(union)) * 100
    return {
        "score": round(jaccard, 2),
        "matched": sorted(list(intersection)),
        "unmatched_jd": sorted(list(jd_lemmas - intersection)),
        "unmatched_resume": sorted(list(resume_lemmas - intersection))
    }

def semantic_sentence_matching_details(jd_text: str, resume_text: str, threshold: float = 0.4) -> dict:
    """
    Splits JD text into sentences and computes cosine similarity
    for each sentence with the entire resume.
    threshold: the minimal similarity for a sentence to be considered matched
    Returns a dict containing:
      - overall_semantic_score: entire JD vs resume similarity (0-100)
      - matched_sentences: list of (sentence, similarity%) above threshold
      - unmatched_sentences: list of (sentence, similarity%) below threshold
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    resume_emb = model.encode(resume_text, convert_to_tensor=True)

    overall_sim = float(util.cos_sim(jd_emb, resume_emb)[0][0])
    overall_score = round(overall_sim * 100, 2)

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

def semantic_similarity(jd_text: str, resume_text: str) -> float:
    """
    Computes overall semantic similarity between JD and resume text,
    returning a percentage (0-100).
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    res_emb = model.encode(resume_text, convert_to_tensor=True)
    sim = float(util.cos_sim(jd_emb, res_emb)[0][0])
    return round(sim * 100, 2)

def combined_matching_score(jd_text: str, resume_text: str, lemma_weight: float = 0.5, sem_weight: float = 0.5) -> float:
    """
    Weighted average of lemma-based similarity & entire-text semantic similarity.
    Returns final score (0-100).
    """
    lemma_info = lemma_based_match_details(jd_text, resume_text)
    lemma_score = lemma_info["score"]

    sem_score = semantic_similarity(jd_text, resume_text)
    combined = (lemma_weight * lemma_score) + (sem_weight * sem_score)
    return round(combined, 2)

########################################
# Streamlit App
########################################

def main():
    st.title("Resume vs. Job Description Matcher")
    st.write("Paste a Job Description and upload a PDF resume. The app will compute matching scores using both lexical (lemma-based) and semantic approaches, plus a final combined score.")

    # 1) Job Description Input
    jd_text = st.text_area("Job Description:", height=200)

    # 2) PDF Upload
    uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])

    if st.button("Process"):
        # Basic validation
        if not jd_text.strip():
            st.error("Please paste a Job Description.")
            return
        if not uploaded_file:
            st.error("Please upload a PDF resume.")
            return

        # Extract raw resume text
        resume_text_raw = extract_text_from_pdf(uploaded_file)
        st.subheader("Raw Resume Text")
        st.text_area("Raw Text", resume_text_raw, height=200)

        # Clean resume text using LLM (if available)
        cleaned_resume_text = clean_resume_with_llm(resume_text_raw)
        st.subheader("Cleaned Resume Text")
        st.text_area("Cleaned Text", cleaned_resume_text, height=200)

        # Parse structured resume details using LLM (if available)
        parsed_details = parse_resume_details(cleaned_resume_text)
        st.subheader("Parsed Resume Details (LLM)")
        st.write(parsed_details)

        # Lemma-based matching
        lemma_details = lemma_based_match_details(jd_text, cleaned_resume_text)
        st.subheader("Lemma-Based Matching Details")
        st.markdown(f"**Lemma-Based Similarity Score:** {lemma_details['score']}%")
        st.markdown("**Matched Lemmas:** " + (", ".join(lemma_details["matched"]) if lemma_details["matched"] else "None"))
        st.markdown("**Unmatched JD Lemmas:** " + (", ".join(lemma_details["unmatched_jd"]) if lemma_details["unmatched_jd"] else "None"))
        st.markdown("**Unmatched Resume Lemmas:** " + (", ".join(lemma_details["unmatched_resume"]) if lemma_details["unmatched_resume"] else "None"))

        # Semantic sentence-level matching
        semantic_info = semantic_sentence_matching_details(jd_text, cleaned_resume_text, threshold=0.4)
        st.subheader("Semantic Matching Details")
        st.markdown(f"**Overall Semantic Similarity Score:** {semantic_info['overall_semantic_score']}%")

        st.markdown("**Matched JD Sentences (above threshold)**")
        if semantic_info["matched_sentences"]:
            for sent, sim_percent in semantic_info["matched_sentences"]:
                st.markdown(f"- {sent} (Score: {sim_percent}%)")
        else:
            st.write("No matched sentences.")

        st.markdown("**Unmatched JD Sentences**")
        if semantic_info["unmatched_sentences"]:
            for sent, sim_percent in semantic_info["unmatched_sentences"]:
                st.markdown(f"- {sent} (Score: {sim_percent}%)")
        else:
            st.write("None.")

        # Final combined score
        combined_score = combined_matching_score(jd_text, cleaned_resume_text, lemma_weight=0.5, sem_weight=0.5)
        st.subheader(f"Combined Matching Score: {combined_score}%")


if __name__ == "__main__":
    main()
