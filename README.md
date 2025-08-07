
---

# Candidate Recommendation Engine

This is a simple web app that recommends the best candidates for a job based on resume relevance to a given job description.

## Features
- Accepts a job description (text input)
- Accepts candidate resumes (PDF, DOCX, or TXT upload)
- Uses sentence-transformers for semantic embeddings
- Computes cosine similarity to rank candidates
- Displays top 5-10 most relevant candidates with name/ID and similarity score
- (Bonus) AI-generated summary for why each candidate is a fit

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Approach & Assumptions
- Embeddings are generated using the `all-MiniLM-L6-v2` model from sentence-transformers.
- Resumes can be uploaded in PDF, DOCX, or TXT format.
- The app runs locally and does not require API keys.

## File Structure
- `app.py`: Main Streamlit app
- `requirements.txt`: Dependencies
- `README.md`: This file

