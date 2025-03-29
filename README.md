# Resume Parser & Job Matching

A project that extracts and summarizes PDF resumes using both **code-based** logic (regex, dictionaries) and **LLM** refinement (Groq LLM). It then compares the finalized resume data against a user-provided **job description** using **lemma-based** and **semantic** matching approaches, yielding detailed matching scores.

---

## Live Demo

**Project Deployment Link**: [Resume Parser App](https://resume-parser-0265.streamlit.app/)

---

## Video Demonstration

Check out the **demo video** here: [Demo Video](https://drive.google.com/file/d/1532enfEnIBFkmdJPWcmF3Q8JPGWMAmrF/view?usp=sharing)

---

## Features

1. **PDF Resume Upload**:  
   Upload any PDF resume file.

2. **Code-Based Parsing**:
   - **Regex** for phone and email.  
   - **List-based** skill detection from known keywords.

3. **LLM Refinement**:
   - A **Groq LLM** verifies partial parsed data, removing incorrect items and producing a **final summary** in subpoints (Education, Experience, Skills, etc.).
   - Fallback text appears if the LLM or API key is unavailable.

4. **Job Description Matching**:
   - **Lemma-Based** (Jaccard / lexical overlap).  
   - **Semantic** (using sentence-transformers to measure embedding similarity).
   - **Combined** final score.

5. **Detailed Outputs**:
   - Raw vs. Cleaned Resume Text  
   - Matched & Unmatched Tokens / Sentences  
   - Final Summaries / Bullet Points

---

## How It Works

1. **User Interaction**:
   - Paste a **job description**.
   - **Upload** a PDF resume.

2. **Partial Parsing**:
   - **Regex** extracts phone/email.  
   - **Keyword** detection finds known skills (e.g., "Python", "Java").

3. **LLM Finalization**:
   - The partial parse plus the raw text is fed into Groq LLM.  
   - The LLM verifies or removes incorrect fields, then produces subheading-based summaries.

4. **Matching with JD**:
   1. **Lemma-Based**: Jaccard overlap between lemmatized tokens from the JD and resume.  
   2. **Semantic**: Overall text similarity plus line-by-line JD comparisons.  
   3. **Combined**: Weighted average, default 50/50.

---

## Local Deployment

1. **Clone** or **Download** the repository.  
2. **Install** the dependencies (pinned in `requirements.txt`):
   ```bash
   pip install -r requirements.txt
   ```
3. **Run** the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. **Open** the URL provided by Streamlit.  
5. **Interact** with the UI to upload a PDF resume and paste a job description.

---

## Project Flow

1. **app.py**:  
   - Presents the Streamlit UI.  
   - Handles user input (resume + JD).  
   - Shows partial parse, LLM finalization, and matching outputs.

2. **requirements.txt**:  
   - Lists pinned versions for stable deployment.

3. **Code & LLM** synergy:
   - The code-based approach ensures partial data extraction without relying solely on the LLM.  
   - The LLM refines that data, producing a final bullet-point summary.

---

## Demo Links

- **Live App**: [Resume Parser App](https://resume-parser-0265.streamlit.app/)  
- **Demo Video**: [Watch on Google Drive](https://drive.google.com/file/d/1532enfEnIBFkmdJPWcmF3Q8JPGWMAmrF/view?usp=sharing)

---

## Known Limitations

- If the **Groq LLM** is unavailable or the key is invalid, you'll see a fallback message.  
- The code-based parse is minimal (phone/email regex, skill dictionary). Extend these methods for deeper extraction.  
- Torch-based dependencies can occasionally cause environment conflicts. See `requirements.txt` for pinned versions.

---

## Contributing

1. **Fork** this repository.  
2. **Create** a new branch for your features/fixes.  
3. **Open** a Pull Request with a clear explanation.

---

## License

This project is open-source under the [MIT License](LICENSE). Feel free to use and adapt it.

---

### Acknowledgements

- **Streamlit** for interactive UI.  
- **PyPDF2** for PDF text extraction.  
- **Groq** for LLM inference.  
- **NLTK & sentence-transformers** for textual processing & semantic matching.

---
