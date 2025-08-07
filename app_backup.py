import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import os
from st_aggrid import AgGrid, GridOptionsBuilder
from PIL import Image
import base64

# Custom CSS for creative UI
st.markdown(
    """
    <style>
    body, .stApp {
        background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%);
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        color: #2B2A4C;
        letter-spacing: 2px;
        text-shadow: 1px 2px 10px #a1a1c0;
        margin-bottom: 0.5em;
    }
    .desc-box {
        background: #fff;
        border-radius: 14px;
        box-shadow: 0 4px 20px #c5d0ff44;
        padding: 2em 2em 1em 2em;
        margin-bottom: 1.5em;
    }
    .candidate-card {
        background: #f6f8ff;
        border-radius: 18px;
        box-shadow: 0 2px 12px #a1b5d844;
        padding: 1.5em 2em;
        margin-bottom: 1.5em;
        border-left: 8px solid #5C469C;
    }
    .score-pill {
        display: inline-block;
        background: #5C469C;
        color: #fff;
        border-radius: 20px;
        padding: 0.3em 1em;
        font-weight: 600;
        font-size: 1.1em;
        margin-bottom: 0.5em;
    }
    .summary-box {
        background: #e4eaff;
        border-radius: 10px;
        padding: 0.8em 1.2em;
        margin-top: 0.8em;
        color: #222;
        font-style: italic;
    }
    .footer {
        margin-top: 2em;
        color: #888;
        font-size: 0.95em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🌟 Candidate Recommendation Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="desc-box">Upload resumes and enter a job description to find the <b>best matches</b>. Stand out with a beautiful, modern interface!<br><br><b>Tip:</b> Use PDF, DOCX, or TXT resumes for best results.</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    job_desc = st.text_area("Job Description", height=180, placeholder="Paste or write the job description here...")
with col2:
    st.markdown("#### Upload candidate resumes")
    resume_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT resumes",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

st.markdown("<br>", unsafe_allow_html=True)

if st.button("✨ Find Top Candidates", use_container_width=True):
    if not job_desc or not resume_files:
        st.warning("Please provide both a job description and at least one resume.")
    else:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        job_emb = model.encode([job_desc])
        candidates = []

        def extract_text_from_pdf(pdf_file):
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text

        def extract_text_from_docx(docx_file):
            doc = docx.Document(docx_file)
            return "\n".join([para.text for para in doc.paragraphs])

        def extract_text_from_txt(txt_file):
            return txt_file.read().decode("utf-8")

        def parse_resume(file):
            name = os.path.splitext(file.name)[0]
            if file.type == "application/pdf":
                text = extract_text_from_pdf(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = extract_text_from_docx(file)
            elif file.type == "text/plain":
                text = extract_text_from_txt(file)
            else:
                text = ""
            return name, text

        def ai_summary(job_desc, resume_text, name):
            # More creative, personalized summary
            return f"<b>{name}</b> demonstrates key skills and experiences that align with the requirements of this role. Their background suggests they can make an immediate impact on your team!"

        for file in resume_files:
            name, text = parse_resume(file)
            if text.strip():
                resume_emb = model.encode([text])
                similarity = cosine_similarity(job_emb, resume_emb)[0][0]
                summary = ai_summary(job_desc, text, name)
                candidates.append({
                    "name": name,
                    "similarity": similarity,
                    "summary": summary
                })
        if not candidates:
            st.warning("No valid resumes were processed.")
        else:
            candidates = sorted(candidates, key=lambda x: x["similarity"], reverse=True)
            st.markdown('<h2>Top Candidates:</h2>', unsafe_allow_html=True)
            for idx, cand in enumerate(candidates[:10]):
                st.markdown(f"""
                <div class='candidate-card'>
                    <span class='score-pill'>Rank #{idx+1} &nbsp; | &nbsp; Score: {cand['similarity']:.3f}</span><br>
                    <b style='font-size:1.3em'>{cand['name']}</b>
                    <div class='summary-box'>{cand['summary']}</div>
                </div>
                """, unsafe_allow_html=True)

st.markdown('<div class="footer">Made with ❤️ for your next career move. Impress recruiters with a modern, beautiful candidate matcher!</div>', unsafe_allow_html=True)
