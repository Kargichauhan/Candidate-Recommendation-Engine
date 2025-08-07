import streamlit as st
from typing import Dict, List
import time

class CandidateExplainer:
    """AI-powered candidate explanation generator"""
    
    def __init__(self):
        # Check if OpenAI API key is available
        self.api_available = False
        try:
            import openai
            api_key = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("openai_api_key")
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
                self.api_available = True
        except Exception as e:
            print(f"OpenAI not available: {e}")
            pass
    
    def generate_explanation(self, job_description: str, resume_text: str, 
                           analysis: Dict, candidate_name: str) -> str:
        """Generate AI explanation for why candidate is a good fit"""
        
        if not self.api_available:
            return self._generate_rule_based_explanation(analysis, candidate_name)
        
        # Create focused prompt for AI
        similarity_score = analysis.get('weighted_similarity', 0)
        skill_overlap = analysis.get('skill_overlap', 0)
        experience_match = analysis.get('experience_score', 0)
        
        # Truncate texts to avoid token limits
        job_summary = job_description[:800]
        resume_summary = resume_text[:1000]
        
        prompt = f"""
        Analyze why this candidate is a good fit for the job position. Be specific and professional.
        
        JOB DESCRIPTION (summary):
        {job_summary}
        
        CANDIDATE RESUME (summary):
        {resume_summary}
        
        ANALYSIS METRICS:
        - Overall Match: {similarity_score:.1%}
        - Technical Skills Match: {skill_overlap:.1%}  
        - Experience Level Match: {experience_match:.1%}
        
        Provide a concise 2-3 sentence explanation highlighting:
        1. Key matching skills or experience that align with the role
        2. Relevant background or achievements
        3. What makes this candidate stand out
        
        Be specific about technical skills, experience level, or industry background. Focus on concrete matches.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.warning(f"AI explanation unavailable: {str(e)}")
            return self._generate_rule_based_explanation(analysis, candidate_name)
    
    def _generate_rule_based_explanation(self, analysis: Dict, candidate_name: str) -> str:
        """Generate rule-based explanation when AI is not available"""
        
        similarity = analysis.get('weighted_similarity', 0)
        skill_overlap = analysis.get('skill_overlap', 0)
        experience_score = analysis.get('experience_score', 0)
        resume_skills = analysis.get('resume_skills', {})
        resume_experience = analysis.get('resume_experience', 0)
        
        explanations = []
        
        # Similarity-based explanation
        if similarity >= 0.8:
            explanations.append("Excellent overall match with strong alignment across multiple criteria.")
        elif similarity >= 0.7:
            explanations.append("Strong candidate with good alignment to job requirements.")
        elif similarity >= 0.6:
            explanations.append("Good fit with solid qualifications for the role.")
        else:
            explanations.append("Reasonable fit with some relevant qualifications.")
        
        # Skills explanation
        if skill_overlap >= 0.6:
            top_skills = []
            for category, skills in resume_skills.items():
                if category != 'all_skills' and skills:
                    top_skills.extend(skills[:2])
            if top_skills:
                skills_text = ', '.join(top_skills[:4])
                explanations.append(f"Strong technical skills including {skills_text}.")
        elif skill_overlap >= 0.3:
            explanations.append("Has relevant technical skills that match some job requirements.")
        elif skill_overlap > 0:
            explanations.append("Shows some technical skills alignment with the position.")
        
        # Experience explanation  
        if resume_experience > 0:
            if experience_score >= 0.8:
                explanations.append(f"Excellent experience level with {resume_experience}+ years in the field.")
            elif experience_score >= 0.6:
                explanations.append(f"Good experience background with {resume_experience} years.")
            else:
                explanations.append(f"Has {resume_experience} years of relevant experience.")
        
        # Ensure we always have at least one explanation
        if not explanations:
            explanations.append("Candidate shows potential for this role.")
        
        return ' '.join(explanations)
    
    def batch_explain(self, job_description: str, candidates: List[Dict]) -> List[Dict]:
        """Generate explanations for multiple candidates with progress tracking"""
        
        if not candidates:
            return []
        
        explained_candidates = []
        
        # Show progress bar for AI explanations
        if self.api_available and len(candidates) > 3:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        for i, candidate in enumerate(candidates):
            if self.api_available and len(candidates) > 3:
                progress = (i + 1) / len(candidates)
                progress_bar.progress(progress)
                status_text.text(f"Generating AI explanation for {candidate.get('name', f'Candidate {i+1}')}...")
            
            explanation = self.generate_explanation(
                job_description,
                candidate['resume_text'],
                candidate,
                candidate.get('name', f'Candidate {i+1}')
            )
            
            candidate_with_explanation = candidate.copy()
            candidate_with_explanation['ai_explanation'] = explanation
            explained_candidates.append(candidate_with_explanation)
            
            # Small delay to avoid rate limiting
            if self.api_available and i < len(candidates) - 1:
                time.sleep(0.5)
        
        # Clear progress indicators
        if self.api_available and len(candidates) > 3:
            progress_bar.empty()
            status_text.empty()
        
        return explained_candidates

def setup_openai_api():
    """Setup OpenAI API key through Streamlit interface"""
    
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = None
    
    # Check if API key is already in secrets
    try:
        if 'OPENAI_API_KEY' in st.secrets:
            st.session_state.openai_api_key = st.secrets['OPENAI_API_KEY']
            return True
    except:
        pass
    
    # Show API key input in sidebar
    with st.sidebar:
        st.markdown("###  AI Explanations")
        api_key = st.text_input(
            "OpenAI API Key (optional)", 
            type="password",
            value=st.session_state.openai_api_key or "",
            help="Enter your OpenAI API key to enable AI-powered candidate explanations. Leave blank to use rule-based explanations."
        )
        
        if api_key and api_key != st.session_state.openai_api_key:
            st.session_state.openai_api_key = api_key
            st.rerun()
        
        if api_key:
            st.success(" AI explanations enabled")
            return True
        else:
            st.info("Rule-based explanations will be used")
            return False

class SimpleExplainer:
    """Fallback explainer that doesn't require OpenAI"""
    
    def __init__(self):
        self.api_available = False
    
    def generate_explanation(self, job_description: str, resume_text: str, 
                           analysis: Dict, candidate_name: str) -> str:
        """Generate simple rule-based explanation"""
        return self._generate_rule_based_explanation(analysis, candidate_name)
    
    def _generate_rule_based_explanation(self, analysis: Dict, candidate_name: str) -> str:
        """Generate rule-based explanation"""
        
        similarity = analysis.get('weighted_similarity', 0)
        skill_overlap = analysis.get('skill_overlap', 0)
        experience_score = analysis.get('experience_score', 0)
        resume_skills = analysis.get('resume_skills', {})
        resume_experience = analysis.get('resume_experience', 0)
        
        explanations = []
        
        # Similarity-based explanation
        if similarity >= 0.8:
            explanations.append("Excellent overall match with strong alignment across multiple criteria.")
        elif similarity >= 0.7:
            explanations.append("Strong candidate with good alignment to job requirements.")
        elif similarity >= 0.6:
            explanations.append("Good fit with solid qualifications for the role.")
        else:
            explanations.append("Reasonable fit with some relevant qualifications.")
        
        # Skills explanation
        if skill_overlap >= 0.5:
            top_skills = []
            for category, skills in resume_skills.items():
                if category != 'all_skills' and skills:
                    top_skills.extend(skills[:2])
            if top_skills:
                skills_text = ', '.join(top_skills[:3])
                explanations.append(f"Strong technical skills including {skills_text}.")
        elif skill_overlap >= 0.2:
            explanations.append("Has relevant technical skills that match some job requirements.")
        
        # Experience explanation  
        if resume_experience > 0:
            if experience_score >= 0.8:
                explanations.append(f"Excellent experience level with {resume_experience}+ years in the field.")
            elif experience_score >= 0.6:
                explanations.append(f"Good experience background with {resume_experience} years.")
            else:
                explanations.append(f"Has {resume_experience} years of relevant experience.")
        
        return ' '.join(explanations) if explanations else "Candidate shows potential for this role."
    
    def batch_explain(self, job_description: str, candidates: List[Dict]) -> List[Dict]:
        """Generate explanations for multiple candidates"""
        
        explained_candidates = []
        
        for candidate in candidates:
            explanation = self.generate_explanation(
                job_description,
                candidate['resume_text'],
                candidate,
                candidate.get('name', 'Candidate')
            )
            
            candidate_with_explanation = candidate.copy()
            candidate_with_explanation['ai_explanation'] = explanation
            explained_candidates.append(candidate_with_explanation)
        
        return explained_candidates