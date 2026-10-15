# 📄 ATS Resume Checker

A Streamlit app that analyzes your resume like an Applicant Tracking System (ATS) would. Upload a resume, optionally paste a job description, and get:

- An **estimated ATS score** (0-100)
- **Category scores**: keywords, formatting, content impact, readability, completeness
- **Strengths and weaknesses**
- **Missing keywords**
- **Prioritised improvements** (High / Medium / Low)
- **Before/after bullet-point rewrites**
- **Job description match %** (when a job description is provided)
- Instant **rule-based checks** (email, phone, section headings, length, bullets, quantified results)
- A downloadable **Markdown report**

Built with [Streamlit](https://streamlit.io) and Google's **Gemini Flash** model.

> **Note:** The score is an AI-based estimate of ATS-friendliness, not the output of any employer's real ATS. Use it as guidance. The overall score is computed in code as a weighted average of the category scores
> (keywords 30%, content 25%, formatting 20%, completeness 15%, readability 10%).

## Supported files

PDF (text-based), DOCX, and TXT. Scanned/image-only PDFs are rejected, because an ATS can't read them either.

## Run locally

1. Get a free Gemini API key at <https://aistudio.google.com/apikey>.
2. Install and run:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

3. Open <http://localhost:8501> and paste your API key in the sidebar.

### Using a key without typing it every time

Pick one:

**Environment variable**
```bash
export GEMINI_API_KEY="your-key"        # Windows (PowerShell): $env:GEMINI_API_KEY="your-key"
```

**Streamlit secrets**: create `.streamlit/secrets.toml` (never commit this file):
```toml
GEMINI_API_KEY = "your-key"
```

The app checks, in order: sidebar input, Streamlit secrets, environment variable.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (the key must **not** be in the repo).
2. Go to <https://share.streamlit.io>, sign in with GitHub and click **Create app**.
3. Select your repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

## Configuration

- **Model**: change it in the sidebar, or edit `DEFAULT_MODEL` in `app.py` (default `gemini-3.8-flash`).
- **Score weights**: edit `CATEGORY_WEIGHTS` in `app.py`.

## Privacy

Resume text is sent to the Gemini API for analysis and is not stored by this app.

## Project structure

```
.
├── app.py             # Streamlit app
├── requirements.txt   # Python dependencies
└── README.md
```
