# Resume Parser

A resume parsing project powered by Groq's Language Processing Unit (LPU) using the `llama-index-llms-groq` integration. This project extracts text from a PDF resume and uses the Groq LLM (specifically the `llama-3.1-8b-instant` model) to parse key details such as full name, contact information, education, work experience, and skills. The code is provided in both Google Colab and Streamlit app formats.

## Live Demo

Try out the live demo of the Streamlit app here: [Resume Parser Demo](https://resume-parser-0265.streamlit.app/)

## Demo Video

Watch the demo video here: [Demo Video](https://drive.google.com/file/d/1yNqaMlMC1mJiDiaVnpCMFv-SEViSthkw/view?usp=sharing)

## Features

- **PDF Text Extraction:** Uses PyPDF2 to extract text from uploaded PDF resumes.
- **Resume Parsing:** Sends the extracted text to the Groq LLM API to extract structured resume details.
- **Structured Output:** Formats the parsed details as bullet points for easy reading.
- **Multiple Interfaces:** Includes a Google Colab notebook version and a Streamlit web app for flexible use.

## Requirements

- Python 3.x
- [llama-index-llms-groq](https://pypi.org/project/llama-index-llms-groq/)
- [llama-index](https://github.com/jerryjliu/llama_index)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [Streamlit](https://streamlit.io/) (only if using the Streamlit app)

Install the required libraries using:

```bash
pip install llama-index-llms-groq llama-index PyPDF2 streamlit
```

## Setup Instructions

1. **Clone or Download the Repository:**
   - Download the source code from the repository.

2. **Configure the Groq API Key:**
   - The API key is hardcoded in the code as:
     ```
     API_KEY=YOUR_API_KEY
     ```
   - Update this key in the code if you have a different one.

3. **Running the Colab Notebook:**
   - Open the provided notebook (e.g., `Groq_Resume_Parser.ipynb`) in [Google Colab](https://colab.research.google.com).
   - Run each cell in order.
   - Upload your PDF resume when prompted to see the parsed output.

4. **Running the Streamlit App:**
   - Save the Streamlit code as `app.py`.
   - Run the app using the command:
     ```bash
     streamlit run app.py
     ```
   - Use the UI to upload your PDF resume and view the parsed details.

## How to Use

### Google Colab Notebook

- **Step 1:** Open the notebook in Colab.
- **Step 2:** Run the installation and import cells.
- **Step 3:** Upload your PDF resume.
- **Step 4:** The notebook will extract the text and call the Groq LLM API to parse the resume.
- **Step 5:** View the parsed details printed in bullet-point format.

### Streamlit App

- **Step 1:** Run `streamlit run app.py` from your terminal.
- **Step 2:** The app displays a user-friendly interface with options.
- **Step 3:** Upload your PDF resume.
- **Step 4:** Click the "Parse Resume" button.
- **Step 5:** The parsed resume details will be displayed on the screen in a structured, bullet-point format.

## Code Overview

- **extract_text_from_pdf:** Extracts text from the uploaded PDF file using PyPDF2.
- **parse_resume_groq:** Sends the resume text to the Groq LLM (using `llama-3.1-8b-instant`) and retrieves the parsed details.
- **format_as_bullets:** Formats the parsed text into bullet points for better readability.
- **Streamlit UI:** Provides a web interface for users to upload resumes and view parsed results.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgements

- **Groq:** For their innovative low-latency LLM solutions.
- **llama-index:** For providing easy-to-use tools for LLM integration.
- **PyPDF2:** For PDF text extraction.
- **Streamlit:** For enabling quick and interactive web apps.
