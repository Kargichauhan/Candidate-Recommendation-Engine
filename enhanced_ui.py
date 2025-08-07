import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import os
import time
import plotly.express as px
import plotly.graph_objects as go

# Import only the analyzer classes to avoid circular imports
try:
    from nlp_analyzer import ResumeAnalyzer
    from ai_explainer import CandidateExplainer, setup_openai_api
    from config import APP_TITLE, APP_DESCRIPTION, EMBEDDING_MODEL, MAX_CANDIDATES_DISPLAY, SIMILARITY_THRESHOLDS
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please make sure all the required files are in your project directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AI Candidate Matcher",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better interactivity
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        animation: headerPulse 3s ease-in-out infinite;
    }
    
    @keyframes headerPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .stMetric {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .candidate-card {
        border: 1px solid #e0e0e0;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .candidate-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }
    
    .upload-section {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background: linear-gradient(135deg, #f8f9ff 0%, #e6f3ff 100%);
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #e6f3ff 0%, #f0f8ff 100%);
        transform: scale(1.02);
    }
    
    .success-message {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        animation: successPulse 2s ease-in-out;
    }
    
    @keyframes successPulse {
        0% { opacity: 0; transform: scale(0.9); }
        50% { opacity: 1; transform: scale(1.05); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    .skill-tag {
        display: inline-block;
        padding: 6px 12px;
        margin: 3px;
        background: linear-gradient(135deg, #e1f5fe 0%, #b3e5fc 100%);
        border-radius: 20px;
        font-size: 0.8em;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .skill-tag:hover {
        transform: scale(1.1);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .progress-animation {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        background-size: 200% 200%;
        animation: gradient 2s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the sentence transformer model with caching"""
    try:
        with st.spinner("🤖 Loading AI model..."):
            model = SentenceTransformer(EMBEDDING_MODEL)
        st.success("✅ AI model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Please check your internet connection and try again.")
        return None

@st.cache_data
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF {pdf_file.name}: {e}")
        return ""

@st.cache_data
def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(docx_file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error reading DOCX {docx_file.name}: {e}")
        return ""

def extract_text_from_txt(txt_file):
    """Extract text from TXT file"""
    try:
        return txt_file.read().decode('utf-8')
    except Exception as e:
        st.error(f"Error reading TXT {txt_file.name}: {e}")
        return ""

def parse_resume(file):
    """Parse resume based on file type"""
    name = os.path.splitext(file.name)[0]
    
    if file.type == "application/pdf":
        text = extract_text_from_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(file)
    elif file.type == "text/plain":
        text = extract_text_from_txt(file)
    else:
        st.warning(f"Unsupported file type: {file.type}")
        return None
    
    if not text.strip():
        st.warning(f"No text extracted from {file.name}")
        return None
    
    return {"name": name, "resume_text": text}

def create_similarity_chart(candidates):
    """Create interactive similarity score visualization"""
    if not candidates:
        return go.Figure()
    
    names = [c.get('name', f'Candidate {i+1}') for i, c in enumerate(candidates[:10])]
    scores = [c.get('weighted_similarity', c.get('similarity_score', 0)) for c in candidates[:10]]
    
    fig = px.bar(
        x=scores,
        y=names,
        orientation='h',
        title='🏆 Top Candidates by Similarity Score',
        labels={'x': 'Similarity Score', 'y': 'Candidate'},
        color=scores,
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_layout(
        height=max(400, len(names) * 40),
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
        xaxis={'range': [0, 1]},
        coloraxis_showscale=False
    )
    
    # Add score labels on bars
    fig.update_traces(
        text=[f'{score:.1%}' for score in scores],
        textposition='inside',
        textfont_color='white'
    )
    
    return fig

def create_skills_distribution_chart(candidates):
    """Create skills distribution chart"""
    if not candidates:
        return go.Figure()
    
    skill_counts = {}
    for candidate in candidates[:10]:
        skills = candidate.get('resume_skills', {})
        for category, skill_list in skills.items():
            if category != 'all_skills' and skill_list:
                category_name = category.replace('_', ' ').title()
                skill_counts[category_name] = skill_counts.get(category_name, 0) + len(skill_list)
    
    if not skill_counts:
        return go.Figure()
    
    categories = list(skill_counts.keys())
    counts = list(skill_counts.values())
    
    fig = px.bar(
        x=categories,
        y=counts,
        title='🔧 Technical Skills Distribution Among Top Candidates',
        labels={'x': 'Skill Category', 'y': 'Number of Skills'},
        color=counts,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        xaxis_tickangle=-45,
        coloraxis_showscale=False
    )
    
    return fig

def create_enhanced_candidate_card(candidate, rank):
    """Create an enhanced candidate display card"""
    similarity = candidate.get('weighted_similarity', candidate.get('similarity_score', 0))
    skill_overlap = candidate.get('skill_overlap', 0)
    experience_score = candidate.get('experience_score', 0)
    
    # Calculate match percentage and status
    match_percentage = similarity * 100
    
    if match_percentage >= SIMILARITY_THRESHOLDS['excellent'] * 100:
        color = "🟢"
        status = "Excellent Match"
    elif match_percentage >= SIMILARITY_THRESHOLDS['good'] * 100:
        color = "🟡"
        status = "Good Match"
    elif match_percentage >= SIMILARITY_THRESHOLDS['fair'] * 100:
        color = "🟠"
        status = "Fair Match"
    else:
        color = "🔴"
        status = "Needs Review"
    
    with st.container():
        # Header with rank and name
        col1, col2, col3 = st.columns([0.5, 3, 1])
        
        with col1:
            st.markdown(f"### #{rank}")
        
        with col2:
            st.markdown(f"### {color} {candidate.get('name', f'Candidate {rank}')}")
            st.markdown(f"**{status}** - {match_percentage:.1f}% overall similarity")
        
        with col3:
            with st.popover("📊 Details"):
                st.metric("Overall Score", f"{similarity:.1%}")
                st.metric("Skills Match", f"{skill_overlap:.1%}")
                st.metric("Experience Fit", f"{experience_score:.1%}")
        
        # Metrics row
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Overall", f"{similarity:.1%}")
        with metric_col2:
            st.metric("Skills", f"{skill_overlap:.1%}")
        with metric_col3:
            exp_years = candidate.get('resume_experience', 0)
            st.metric("Experience", f"{exp_years} years")
        with metric_col4:
            education = candidate.get('resume_education', 'Not specified')
            st.metric("Education", education.title())
        
        # AI Explanation
        if 'ai_explanation' in candidate and candidate['ai_explanation']:
            st.info(f"🤖 **AI Analysis**: {candidate['ai_explanation']}")
        
        # Skills display
        resume_skills = candidate.get('resume_skills', {})
        if resume_skills:
            st.write("**🔧 Technical Skills:**")
            
            all_skills = []
            for category, skills in resume_skills.items():
                if category != 'all_skills' and skills:
                    all_skills.extend(skills[:3])
            
            if all_skills:
                skills_html = ""
                colors = ['#e1f5fe', '#f3e5f5', '#e8f5e8', '#fff3e0', '#fce4ec']
                for i, skill in enumerate(all_skills[:8]):
                    bg_color = colors[i % len(colors)]
                    skills_html += f'<span class="skill-tag" style="background-color: {bg_color};">{skill}</span> '
                
                st.markdown(skills_html, unsafe_allow_html=True)
                
                if len(all_skills) > 8:
                    st.caption(f"... and {len(all_skills) - 8} more skills")
        
        # Expandable resume section
        with st.expander("📄 View Full Resume"):
            st.text_area(
                "Resume Content", 
                candidate.get('resume_text', 'No resume content available'), 
                height=200, 
                disabled=True,
                key=f"resume_text_{rank}"
            )
        
        st.divider()

def display_analytics_dashboard(candidates, job_description):
    """Display comprehensive analytics dashboard"""
    if not candidates:
        st.warning("No candidates to analyze.")
        return
    
    st.markdown("## 📊 Recruitment Analytics")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = sum([c.get('weighted_similarity', c.get('similarity_score', 0)) for c in candidates]) / len(candidates)
        st.metric("Average Match", f"{avg_score:.1%}")
    
    with col2:
        threshold = SIMILARITY_THRESHOLDS['good']
        strong_matches = len([c for c in candidates if c.get('weighted_similarity', c.get('similarity_score', 0)) >= threshold])
        st.metric("Strong Matches", f"{strong_matches}")
    
    with col3:
        total_candidates = len(candidates)
        st.metric("Total Candidates", f"{total_candidates}")
    
    with col4:
        if candidates:
            best_match = max([c.get('weighted_similarity', c.get('similarity_score', 0)) for c in candidates])
            st.metric("Best Match", f"{best_match:.1%}")
    
    # Charts in tabs
    tab1, tab2 = st.tabs(["📈 Rankings", "🎯 Skills Analysis"])
    
    with tab1:
        fig = create_similarity_chart(candidates)
        if fig.data:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for similarity chart")
    
    with tab2:
        fig = create_skills_distribution_chart(candidates)
        if fig.data:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No skills data available for analysis")

def display_job_analysis_sidebar(job_description, analyzer):
    """Display job analysis in sidebar"""
    if not job_description.strip():
        return
    
    with st.sidebar:
        st.markdown("### 📋 Job Requirements Analysis")
        
        job_skills = analyzer.extract_skills(job_description)
        job_experience = analyzer.extract_experience_years(job_description)
        job_education = analyzer.extract_education_level(job_description)
        
        # Required experience
        if job_experience > 0:
            st.metric("Required Experience", f"{job_experience} years")
        
        # Education requirement
        if job_education != 'other':
            st.metric("Education Level", job_education.title())
        
        # Key skills
        if job_skills:
            st.write("**Required Skills:**")
            all_skills = []
            for category, skills in job_skills.items():
                if category != 'all_skills' and skills:
                    all_skills.extend(skills[:3])
            
            if all_skills:
                for skill in all_skills[:8]:
                    st.caption(f"• {skill}")
                if len(all_skills) > 8:
                    st.caption(f"... and {len(all_skills) - 8} more")

def real_time_candidate_preview(candidate_data):
    """Real-time candidate data preview as user types"""
    if not candidate_data.strip():
        return
    
    with st.expander("👀 Real-time Preview", expanded=True):
        # Quick stats
        word_count = len(candidate_data.split())
        char_count = len(candidate_data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Word Count", word_count)
        with col2:
            st.metric("Character Count", char_count)
        with col3:
            quality = "Good" if word_count > 100 else "Needs more detail"
            st.metric("Quality", quality)
        
        # Quick skill detection
        detected_skills = []
        common_skills = ['python', 'java', 'react', 'aws', 'sql', 'docker', 'kubernetes']
        
        for skill in common_skills:
            if skill.lower() in candidate_data.lower():
                detected_skills.append(skill.title())
        
        if detected_skills:
            st.write("**🔍 Detected Skills:**")
            skills_html = ""
            for skill in detected_skills:
                skills_html += f'<span class="skill-tag">{skill}</span>'
            st.markdown(skills_html, unsafe_allow_html=True)

def display_interactive_samples():
    """Interactive sample job descriptions with preview"""
    with st.expander("🎮 Interactive Job Description Samples", expanded=False):
        samples = {
            "🤖 Machine Learning Engineer": """We are seeking a Machine Learning Engineer with 3+ years of experience to join our AI team.

Key Requirements:
- Master's degree in Computer Science, Data Science, or related field
- 3+ years of experience in machine learning and deep learning
- Proficiency in Python, TensorFlow, PyTorch, and scikit-learn
- Experience with cloud platforms (AWS, GCP, Azure)
- Strong knowledge of data preprocessing, feature engineering, and model evaluation

Responsibilities:
- Design and implement machine learning models
- Collaborate with data scientists and software engineers
- Deploy models to production environments""",
            
            "💻 Full Stack Developer": """We are looking for a Full Stack Developer with strong experience in modern web technologies.

Requirements:
- Bachelor's degree in Computer Science or equivalent experience
- 2+ years of full-stack development experience
- Proficiency in JavaScript, React, Node.js
- Experience with databases (PostgreSQL, MongoDB)
- Knowledge of RESTful APIs and GraphQL

Nice to have:
- TypeScript experience
- Docker and Kubernetes knowledge""",
            
            "📊 Data Scientist": """Join our data science team to drive insights from complex datasets.

Qualifications:
- PhD or Master's in Statistics, Mathematics, Computer Science
- 4+ years of experience in data science and analytics
- Expert-level Python and R programming skills
- Experience with pandas, numpy, scikit-learn, matplotlib
- Strong statistical analysis and hypothesis testing skills"""
        }
        
        for title, description in samples.items():
            if st.button(f"✨ Use {title}", key=f"sample_{title}"):
                st.session_state.job_description = description
                st.success(f"✅ Loaded {title} sample!")
                st.rerun()

def main():
    # Enhanced app header
    st.markdown(f"""
    <div class="main-header">
        <h1>🎯 {APP_TITLE}</h1>
        <p>{APP_DESCRIPTION}</p>
        <p><em>✨ Enhanced with Interactive Features!</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'processed_candidates' not in st.session_state:
        st.session_state.processed_candidates = []
    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""
    if 'resume_inputs' not in st.session_state:
        st.session_state.resume_inputs = [{"name": "", "text": ""}]
    
    # Setup AI explainer
    setup_openai_api()
    
    # Initialize components
    model = load_model()
    if not model:
        st.stop()
    
    analyzer = ResumeAnalyzer()
    explainer = CandidateExplainer()
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Interactive Controls")
        
        # File upload method selection
        st.subheader("📁 Input Method")
        upload_method = st.radio(
            "Choose input method:",
            ["📁 Upload Files", "✍️ Text Input"],
            help="Upload resume files or paste resume text directly"
        )
        
        # Settings
        st.subheader("🎮 Settings")
        max_candidates = st.slider(
            "Max candidates to show", 
            min_value=5, 
            max_value=20, 
            value=MAX_CANDIDATES_DISPLAY,
            help="Maximum number of candidates to display in results"
        )
        
        show_analytics = st.checkbox("📊 Show Analytics Dashboard", value=True)
        include_ai_explanations = st.checkbox("🤖 Include AI Explanations", value=True)
        use_weighted_scoring = st.checkbox("⚖️ Advanced Weighted Scoring", value=True)
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.processed_candidates = []
            st.session_state.job_description = ""
            st.session_state.resume_inputs = [{"name": "", "text": ""}]
            st.success("All data cleared!")
            st.rerun()
        
        if st.button("🎲 Load Demo", use_container_width=True):
            st.session_state.job_description = """We are seeking a Machine Learning Engineer with 3+ years of experience.
            
Key Requirements:
- Master's degree in Computer Science
- 3+ years of ML experience
- Python, TensorFlow, PyTorch
- AWS/GCP experience"""
            
            st.session_state.resume_inputs = [
                {"name": "Alice Johnson", "text": "ML Engineer with 4 years experience. PhD in CS. Expert in Python, TensorFlow, AWS."},
                {"name": "Bob Wilson", "text": "Software Engineer with 2 years experience. BS in CS. Python, React, Node.js, Docker."}
            ]
            st.success("Demo data loaded!")
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📝 Job & Candidates", "🚀 Analysis", "📊 Results"])
    
    with tab1:
        # Job description
        st.markdown("## 📝 Job Description")
        display_interactive_samples()
        
        job_description = st.text_area(
            "Enter the job description:",
            height=250,
            placeholder="Paste the job description here...",
            value=st.session_state.job_description,
            help="Be specific about requirements for better matching"
        )
        
        if job_description != st.session_state.job_description:
            st.session_state.job_description = job_description
            st.session_state.processed_candidates = []
        
        # Real-time job analysis
        if job_description.strip():
            display_job_analysis_sidebar(job_description, analyzer)
        
        st.markdown("---")
        
        # Resume input
        st.markdown("## 📄 Candidate Resumes")
        candidates = []
        
        if "📁 Upload Files" in upload_method:
            st.markdown("""
            <div class="upload-section">
                <h4>📁 Upload Resume Files</h4>
                <p>Drag and drop multiple files or click to browse</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_files = st.file_uploader(
                "Choose files",
                type=['pdf', 'docx', 'txt'],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} files uploaded!")
                for file in uploaded_files:
                    candidate = parse_resume(file)
                    if candidate and candidate['resume_text'].strip():
                        candidates.append(candidate)
        
        else:  # Text Input
            for i, resume_input in enumerate(st.session_state.resume_inputs):
                st.markdown(f"**👤 Candidate {i+1}:**")
                
                col_name, col_remove = st.columns([4, 1])
                with col_name:
                    name = st.text_input(
                        f"Name",
                        value=resume_input["name"],
                        key=f"name_{i}",
                        placeholder="Enter candidate name..."
                    )
                
                with col_remove:
                    if len(st.session_state.resume_inputs) > 1:
                        if st.button("🗑️", key=f"remove_{i}"):
                            st.session_state.resume_inputs.pop(i)
                            st.rerun()
                
                text = st.text_area(
                    f"Resume Content",
                    value=resume_input["text"],
                    height=150,
                    key=f"text_{i}",
                    placeholder="Paste resume content here..."
                )
                
                # Real-time preview
                if text.strip():
                    real_time_candidate_preview(text)
                
                st.session_state.resume_inputs[i] = {"name": name or f"Candidate {i+1}", "text": text}
                
                if text.strip():
                    candidates.append({
                        "name": name or f"Candidate {i+1}",
                        "resume_text": text
                    })
                
                st.divider()
            
            # Add/Clear buttons
            col_add, col_clear = st.columns(2)
            with col_add:
                if st.button("➕ Add Candidate"):
                    st.session_state.resume_inputs.append({"name": "", "text": ""})
                    st.rerun()
            with col_clear:
                if st.button("🗑️ Clear All"):
                    st.session_state.resume_inputs = [{"name": "", "text": ""}]
                    st.rerun()
    
    with tab2:
        st.markdown("## 🚀 Analysis")
        
        if candidates and job_description.strip():
            st.info(f"📊 Ready to analyze **{len(candidates)} candidates**")
            
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Generate embeddings
                    status_text.text("🧠 Generating embeddings...")
                    progress_bar.progress(0.2)
                    
                    job_embedding = model.encode([job_description])[0]
                    resume_texts = [c['resume_text'] for c in candidates]
                    resume_embeddings = model.encode(resume_texts)
                    
                    # Analyze candidates
                    status_text.text("🔍 Analyzing candidates...")
                    progress_bar.progress(0.5)
                    
                    enhanced_candidates = []
                    
                    for i, candidate in enumerate(candidates):
                        similarity = cosine_similarity([job_embedding], [resume_embeddings[i]])[0][0]
                        
                        if use_weighted_scoring:
                            analysis = analyzer.calculate_weighted_similarity(
                                job_description, candidate['resume_text'], similarity
                            )
                        else:
                            analysis = {
                                'base_similarity': similarity,
                                'weighted_similarity': similarity,
                                'skill_overlap': 0,
                                'experience_score': 1,
                                'education_score': 1,
                                'resume_skills': {},
                                'resume_experience': 0,
                                'resume_education': 'not analyzed'
                            }
                        
                        enhanced_candidate = candidate.copy()
                        enhanced_candidate.update(analysis)
                        enhanced_candidates.append(enhanced_candidate)
                    
                    # Sort candidates
                    enhanced_candidates.sort(key=lambda x: x['weighted_similarity'], reverse=True)
                    
                    # Generate AI explanations
                    if include_ai_explanations:
                        status_text.text("🤖 Generating explanations...")
                        progress_bar.progress(0.8)
                        enhanced_candidates = explainer.batch_explain(job_description, enhanced_candidates)
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Analysis complete!")
                    
                    st.session_state.processed_candidates = enhanced_candidates
                    st.balloons()
                    st.success(f"🎉 Analysis complete! Check the Results tab.")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.info("📝 Please add job description and candidates in the first tab.")
    
    with tab3:
        st.markdown("## 📊 Results")
        
        if st.session_state.processed_candidates:
            candidates_to_show = st.session_state.processed_candidates[:max_candidates]
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                best_match = candidates_to_show[0]['weighted_similarity']
                st.metric("🥇 Best Match", f"{best_match:.1%}")
            
            with col2:
                avg_score = np.mean([c['weighted_similarity'] for c in candidates_to_show])
                st.metric("📊 Average Score", f"{avg_score:.1%}")
            
            with col3:
                strong_matches = len([c for c in candidates_to_show if c['weighted_similarity'] >= 0.7])
                st.metric("🎯 Strong Matches", f"{strong_matches}")
            
            with col4:
                total_analyzed = len(st.session_state.processed_candidates)
                st.metric("👥 Total Analyzed", f"{total_analyzed}")
            
            # Display options
            st.markdown("### 🎮 Display Options")
            
            display_cols = st.columns(3)
            
            with display_cols[0]:
                view_mode = st.selectbox(
                    "View Mode:",
                    ["📋 Detailed Cards", "📊 Analytics Dashboard", "📈 Quick Table"]
                )
            
            with display_cols[1]:
                sort_by = st.selectbox(
                    "Sort by:",
                    ["Overall Score", "Skills Match", "Experience"]
                )
            
            with display_cols[2]:
                filter_threshold = st.slider(
                    "Min score:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1
                )
            
            # Apply filters and sorting
            filtered_candidates = [c for c in candidates_to_show if c['weighted_similarity'] >= filter_threshold]
            
            if sort_by == "Skills Match":
                filtered_candidates.sort(key=lambda x: x.get('skill_overlap', 0), reverse=True)
            elif sort_by == "Experience":
                filtered_candidates.sort(key=lambda x: x.get('resume_experience', 0), reverse=True)
            
            st.markdown("---")
            
            # Display results based on view mode
            if view_mode == "📊 Analytics Dashboard":
                display_analytics_dashboard(filtered_candidates, job_description)
            
            elif view_mode == "📈 Quick Table":
                # Quick table view
                table_data = []
                for i, candidate in enumerate(filtered_candidates):
                    table_data.append({
                        'Rank': i + 1,
                        'Name': candidate.get('name', f'Candidate {i+1}'),
                        'Overall Score': f"{candidate.get('weighted_similarity', 0):.1%}",
                        'Skills Match': f"{candidate.get('skill_overlap', 0):.1%}",
                        'Experience': f"{candidate.get('resume_experience', 0)} years",
                        'Education': candidate.get('resume_education', 'Not specified').title(),
                        'Status': '🟢 Excellent' if candidate.get('weighted_similarity', 0) >= 0.8 else 
                                 '🟡 Good' if candidate.get('weighted_similarity', 0) >= 0.7 else '🟠 Fair'
                    })
                
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            
            else:  # Detailed Cards (default)
                st.markdown("### 👤 Detailed Candidate Rankings")
                
                for i, candidate in enumerate(filtered_candidates):
                    create_enhanced_candidate_card(candidate, i + 1)
            
            # Export section
            st.markdown("---")
            st.markdown("### 📥 Export Results")
            
            export_cols = st.columns(3)
            
            with export_cols[0]:
                if st.button("📊 Export CSV", use_container_width=True):
                    export_data = []
                    for i, candidate in enumerate(filtered_candidates):
                        export_data.append({
                            'Rank': i + 1,
                            'Name': candidate.get('name', f'Candidate {i+1}'),
                            'Overall_Score': f"{candidate.get('weighted_similarity', 0):.1%}",
                            'Skills_Match': f"{candidate.get('skill_overlap', 0):.1%}",
                            'Experience_Years': candidate.get('resume_experience', 0),
                            'Education': candidate.get('resume_education', 'Not specified'),
                            'AI_Explanation': candidate.get('ai_explanation', 'N/A')
                        })
                    
                    df = pd.DataFrame(export_data)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"candidate_rankings_{int(time.time())}.csv",
                        mime="text/csv"
                    )
            
            with export_cols[1]:
                if st.button("📄 Generate Report", use_container_width=True):
                    # Generate comprehensive report
                    report_lines = []
                    report_lines.append("🎯 CANDIDATE RECOMMENDATION REPORT")
                    report_lines.append("=" * 50)
                    report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    report_lines.append(f"Total Candidates: {len(st.session_state.processed_candidates)}")
                    report_lines.append("")
                    
                    # Add job summary
                    report_lines.append("📋 JOB SUMMARY:")
                    job_summary = job_description[:300] + "..." if len(job_description) > 300 else job_description
                    report_lines.append(job_summary)
                    report_lines.append("")
                    
                    # Add top candidates
                    report_lines.append("🏆 TOP CANDIDATES:")
                    report_lines.append("-" * 30)
                    
                    for i, candidate in enumerate(filtered_candidates[:5]):
                        report_lines.append(f"{i+1}. {candidate.get('name', f'Candidate {i+1}')}")
                        report_lines.append(f"   📊 Overall Score: {candidate.get('weighted_similarity', 0):.1%}")
                        report_lines.append(f"   🔧 Skills Match: {candidate.get('skill_overlap', 0):.1%}")
                        report_lines.append(f"   💼 Experience: {candidate.get('resume_experience', 0)} years")
                        if 'ai_explanation' in candidate:
                            report_lines.append(f"   🤖 AI Analysis: {candidate['ai_explanation']}")
                        report_lines.append("")
                    
                    report_text = "\n".join(report_lines)
                    
                    st.download_button(
                        label="Download Report",
                        data=report_text,
                        file_name=f"recruitment_report_{int(time.time())}.txt",
                        mime="text/plain"
                    )
            
            with export_cols[2]:
                if st.button("🔄 Re-analyze", use_container_width=True):
                    st.session_state.processed_candidates = []
                    st.success("Ready for re-analysis!")
                    st.rerun()
        
        else:
            # Enhanced empty state
            st.markdown("""
            <div style="text-align: center; padding: 3rem;">
                <h3>🎯 Ready to Find Perfect Candidates?</h3>
                <p>Complete the analysis in the 'Analysis' tab to see detailed results here.</p>
                <p><em>Your results will include interactive charts, detailed candidate cards, and export options!</em></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced footer
    st.markdown("---")
    
    footer_cols = st.columns([2, 1, 2])
    
    with footer_cols[0]:
        st.markdown("""
        **🎯 AI-Powered Candidate Recommendation Engine**  
        *Enhanced with Interactive Features*
        """)
    
    with footer_cols[1]:
        if st.button(" About"):
            st.info("""
            This enhanced recommendation engine uses:
            -  Advanced AI embeddings
            -  Skills extraction & matching  
            -  Interactive visualizations
            -  AI-powered explanations
            -  Real-time analysis
            """)
    
    with footer_cols[2]:
        st.markdown("""
        <div style='text-align: right; color: #666; font-size: 0.8em;'>
            Built with Streamlit • copyright@kargichauhan<br>
            <em>Making recruitment smarter, faster, and more interactive</em>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()