import streamlit as st
import PyPDF2
import io
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage

# Hardcode your Groq API key here
GROQ_API_KEY = "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"

# ------------- Helper Functions ------------- #
def extract_text_from_pdf(pdf_file) -> str:
    """Extracts text from the uploaded PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    all_text = []
    for page in pdf_reader.pages:
        text = page.extract_text() or ""
        all_text.append(text)
    return "\n".join(all_text)

def parse_resume_groq(resume_text: str, model_name: str) -> str:
    """
    Uses the Groq LLM (via llama_index) to parse resume details.
    Returns the model's generated string.
    """
    system_instruction = (
        "You are an AI assistant that extracts structured data from resumes "
        "and responds concisely with bullet points or short paragraphs."
    )
    user_prompt = f"""
    Please extract the following details from the resume:

    - Full Name
    - Contact Information
    - Education
    - Work Experience
    - Skills

    Resume:
    {resume_text}

    Make sure to respond ONLY with the key details under each heading.
    """
    messages = [
        ChatMessage(role="system", content=system_instruction),
        ChatMessage(role="user", content=user_prompt),
    ]
    llm = Groq(model=model_name, api_key=GROQ_API_KEY)
    response = llm.chat(messages)
    # Convert the response to a string
    return str(response)

def format_as_bullets(parsed_text: str) -> str:
    """
    Formats the parsed text into bullet points.
    """
    lines = parsed_text.strip().splitlines()
    bullets = []
    for line in lines:
        line = line.strip()
        if line:
            bullets.append(f"- {line}")
    return "\n".join(bullets)

# ------------- Streamlit UI ------------- #
def main():
    st.set_page_config(page_title="Groq-Powered Resume Parser", layout="centered")
    
    # Custom CSS for a clean UI
    st.markdown(
        """
        <style>
        body {
            background-color: #f9f9f9;
        }
        .main > div {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
        h1 {
            color: #354052;
            text-align: center;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("Groq-Powered Resume Parser")
    st.write(
        "Upload a PDF resume, and let the **Groq LLM** parse out key details "
        "like name, contact info, education, work experience, and skills."
    )

    # Sidebar for model configuration
    st.sidebar.header("Configuration")
    model_name = st.sidebar.text_input(
        label="Model Name",
        value="llama-3.1-8b-instant",
        help="Enter the Groq LLM model you wish to use (e.g., llama-3.1-8b-instant)."
    )
    st.sidebar.write("---")

    # File uploader for PDF resume
    uploaded_file = st.file_uploader(
        "Upload your PDF resume",
        type=["pdf"],
        help="Upload a single PDF file containing the resume."
    )
    parse_button = st.button("Parse Resume")

    if parse_button and uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            resume_text = extract_text_from_pdf(uploaded_file)

        st.subheader("Extracted Resume Text")
        st.text_area("Preview", resume_text, height=200)

        st.subheader("Parsed Resume Details")
        with st.spinner("Parsing resume with Groq LLM..."):
            try:
                raw_response = parse_resume_groq(resume_text, model_name)
                bullet_response = format_as_bullets(raw_response)
                st.markdown(bullet_response)
            except Exception as e:
                st.error(f"Error during Groq API inference: {e}")
    elif parse_button and uploaded_file is None:
        st.error("Please upload a PDF resume first.")

if __name__ == "__main__":
    main()
