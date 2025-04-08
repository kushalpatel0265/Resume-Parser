import streamlit as st
import PyPDF2, re, io, nltk, numpy as np

# ────────── NLTK setup ──────────
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ────────── Sentence‑transformers ──────────
from sentence_transformers import SentenceTransformer, util
EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ────────── TF‑IDF ──────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ────────── Optional Groq LLM ──────────
try:
    from llama_index.llms.groq import Groq
    from llama_index.core.llms import ChatMessage
    GROQ_API_KEY = "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"        # <— add key or leave blank
    USE_GROQ = bool(GROQ_API_KEY)
except ImportError:
    USE_GROQ = False

# ────────── Config / Weights ──────────
WEIGHT_TFIDF  = 0.30   # 30 %
WEIGHT_SEMANT = 0.70   # 70 %
SKILL_SYNONYMS = {"js": "javascript", "py": "python", "node": "node.js"}

# ────────── Helper functions ──────────
def pdf_to_text(pdf_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(p.extract_text() or "" for p in reader.pages)

def apply_synonyms(txt: str) -> str:
    out = txt.lower()
    for short, full in SKILL_SYNONYMS.items():
        out = re.sub(rf"\b{re.escape(short)}\b", full, out)
    return out

PHONE_RGX = r"\+?\d[\d\s().-]{7,}\d"
MAIL_RGX  = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

def code_parse(txt: str) -> dict:
    return {
        "phones": list(set(re.findall(PHONE_RGX, txt))),
        "emails": list(set(re.findall(MAIL_RGX, txt)))
    }

def llm_refine(raw: str, partial: dict) -> str:
    if not USE_GROQ:
        return "(LLM disabled)\n" + str(partial)
    system = "You finalize resume data, removing incorrect info and summarizing."
    user   = f"""Partial parse:
Phones:{partial['phones']}
Emails:{partial['emails']}

Resume:
{raw}

Return bullet summary (Education, Experience, Skills)."""
    try:
        llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
        resp = llm.chat([ChatMessage(role="system", content=system),
                         ChatMessage(role="user",   content=user)])
        return str(resp)
    except Exception as e:
        return f"(LLM error) {e}\nPartial parse:{partial}"

def tfidf_score(jd: str, res: str) -> float:
    vec = TfidfVectorizer()
    tf  = vec.fit_transform([jd, res])
    return round(cosine_similarity(tf[0], tf[1])[0][0]*100, 2)

def chunk_sem_score(jd: str, res: str) -> float:
    jd_lines = [l.strip() for l in jd.split("\n") if l.strip()]
    res_pars = [p.strip() for p in res.split("\n\n") if p.strip()]
    if not jd_lines or not res_pars:
        return 0.0
    res_emb = EMB_MODEL.encode(res_pars, convert_to_tensor=True)
    sims=[]
    for line in jd_lines:
        line_emb = EMB_MODEL.encode(line, convert_to_tensor=True)
        sims.append(float(util.cos_sim(line_emb, res_emb)[0].max()))
    return round(np.mean(sims)*100, 2)

def final_score(tfidf_val: float, sem_val: float) -> float:
    return min(round(tfidf_val*WEIGHT_TFIDF + sem_val*WEIGHT_SEMANT, 2), 100.0)

# ────────── Streamlit UI ──────────
st.title("Resume ↔ Job‑Description Matcher (TF‑IDF + Semantic)")

jd_text = st.text_area("Paste Job Description", height=180)
pdf_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if st.button("Process"):
    if not jd_text.strip():
        st.error("Please paste a Job Description.")
        st.stop()
    if not pdf_file:
        st.error("Please upload a PDF resume.")
        st.stop()

    raw = pdf_to_text(pdf_file.read())
    st.subheader("Raw Resume Text")
    st.text_area("", raw, height=200)

    # synonym replacement
    resume_syn = apply_synonyms(raw)

    # code parse
    partial = code_parse(resume_syn)
    st.subheader("Code‑Based Parse"); st.json(partial)

    # LLM refine
    st.subheader("LLM Summary"); st.write(llm_refine(raw, partial))

    # Scores
    tf_val  = tfidf_score(jd_text, resume_syn)
    sem_val = chunk_sem_score(jd_text, resume_syn)
    final   = final_score(tf_val, sem_val)

    st.subheader("Scores")
    st.write(f"TF‑IDF lexical overlap : **{tf_val}%**")
    st.write(f"Chunk‑level semantic   : **{sem_val}%**")
    st.subheader(f"Final Weighted Score  : **{final}%**")
