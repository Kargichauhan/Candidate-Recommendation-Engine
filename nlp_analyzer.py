import re
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
from config import TECH_SKILLS_DB

class ResumeAnalyzer:
    """Advanced resume analyzer with NLP capabilities"""
    
    def __init__(self):
        self.tech_skills = TECH_SKILLS_DB
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract technical skills from text"""
        text_lower = text.lower()
        found_skills = {}
        all_found_skills = []
        
        for category, skills in self.tech_skills.items():
            found_skills[category] = []
            for skill in skills:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(skill.replace('-', r'[-\s]?')) + r'\b'
                if re.search(pattern, text_lower):
                    found_skills[category].append(skill)
                    all_found_skills.append(skill)
        
        # Remove empty categories
        found_skills = {k: v for k, v in found_skills.items() if v}
        found_skills['all_skills'] = all_found_skills
        
        return found_skills
    
    def extract_experience_years(self, text: str) -> int:
        """Extract years of experience from resume"""
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*in',
            r'experience.*?(\d+)\+?\s*years?',
            r'(\d+)\+?\s*year\s*experience',
            r'(\d+)\+?\s*yrs?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*yrs?\s*in',
        ]
        
        years = []
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            years.extend([int(match) for match in matches if match.isdigit() and int(match) <= 50])
        
        return max(years) if years else 0
    
    def extract_education_level(self, text: str) -> str:
        """Extract highest education level"""
        text_lower = text.lower()
        
        education_levels = [
            ('phd', ['ph.d', 'phd', 'doctorate', 'doctoral', 'doctor of philosophy']),
            ('masters', ['master', 'ms', 'm.s.', 'msc', 'm.sc.', 'mba', 'm.b.a.', 'ma', 'm.a.', 'meng', 'm.eng']),
            ('bachelors', ['bachelor', 'bs', 'b.s.', 'ba', 'b.a.', 'bsc', 'b.sc.', 'be', 'b.e.', 'btech', 'b.tech']),
            ('associates', ['associate', 'aa', 'a.a.', 'as', 'a.s.']),
        ]
        
        for level, keywords in education_levels:
            for keyword in keywords:
                if keyword in text_lower:
                    return level
        
        return 'other'
    
    def extract_companies(self, text: str) -> List[str]:
        """Extract company names (simple heuristic)"""
        lines = text.split('\n')
        companies = []
        
        company_indicators = ['inc', 'corp', 'llc', 'ltd', 'company', 'technologies', 'systems', 'solutions']
        
        for line in lines:
            line = line.strip()
            if any(indicator in line.lower() for indicator in company_indicators):
                # Clean up the line
                clean_line = re.sub(r'[^\w\s&.-]', '', line)
                if len(clean_line.split()) <= 5 and len(clean_line) > 3:
                    companies.append(clean_line.strip())
        
        return companies[:5]  # Return top 5
    
    def calculate_skill_overlap(self, job_skills: Dict, resume_skills: Dict) -> float:
        """Calculate skill overlap percentage"""
        job_all_skills = set(job_skills.get('all_skills', []))
        resume_all_skills = set(resume_skills.get('all_skills', []))
        
        if not job_all_skills:
            return 0.0
        
        overlap = len(job_all_skills.intersection(resume_all_skills))
        return overlap / len(job_all_skills)
    
    def calculate_weighted_similarity(self, job_desc: str, resume: str, 
                                   base_similarity: float) -> Dict[str, float]:
        """Calculate weighted similarity based on different factors"""
        
        # Extract features from both texts
        job_skills = self.extract_skills(job_desc)
        resume_skills = self.extract_skills(resume)
        
        job_experience = self.extract_experience_years(job_desc)
        resume_experience = self.extract_experience_years(resume)
        
        job_education = self.extract_education_level(job_desc)
        resume_education = self.extract_education_level(resume)
        
        # Calculate skill overlap score
        skill_overlap = self.calculate_skill_overlap(job_skills, resume_skills)
        
        # Calculate experience match score
        experience_score = 1.0
        if job_experience > 0:
            if resume_experience >= job_experience:
                experience_score = 1.0
            elif resume_experience >= job_experience * 0.7:
                experience_score = 0.8
            elif resume_experience >= job_experience * 0.5:
                experience_score = 0.6
            else:
                experience_score = 0.4
        
        # Education level matching
        education_levels = {'phd': 4, 'masters': 3, 'bachelors': 2, 'associates': 1, 'other': 0}
        job_edu_level = education_levels.get(job_education, 0)
        resume_edu_level = education_levels.get(resume_education, 0)
        
        if resume_edu_level >= job_edu_level:
            education_score = 1.0
        elif resume_edu_level >= job_edu_level - 1:
            education_score = 0.8
        else:
            education_score = 0.6
        
        # Weighted final score
        weighted_similarity = (
            base_similarity * 0.5 +      # Base semantic similarity
            skill_overlap * 0.3 +        # Technical skills match
            experience_score * 0.15 +    # Experience level match
            education_score * 0.05       # Education match
        )
        
        return {
            'base_similarity': base_similarity,
            'skill_overlap': skill_overlap,
            'experience_score': experience_score,
            'education_score': education_score,
            'weighted_similarity': min(weighted_similarity, 1.0),  # Cap at 1.0
            'job_skills': job_skills,
            'resume_skills': resume_skills,
            'job_experience': job_experience,
            'resume_experience': resume_experience,
            'job_education': job_education,
            'resume_education': resume_education
        }
    
    def get_skill_categories_summary(self, skills: Dict) -> str:
        """Get a summary of skill categories"""
        summaries = []
        for category, skill_list in skills.items():
            if category != 'all_skills' and skill_list:
                category_name = category.replace('_', ' ').title()
                if len(skill_list) <= 3:
                    summaries.append(f"{category_name}: {', '.join(skill_list)}")
                else:
                    summaries.append(f"{category_name}: {', '.join(skill_list[:3])} (+{len(skill_list)-3} more)")
        
        return '; '.join(summaries) if summaries else "No specific technical skills detected"
    
    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information from resume"""
        contact_info = {}
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            contact_info['email'] = emails[0]
        
        # Phone pattern
        phone_pattern = r'(\+?1?[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        phones = re.findall(phone_pattern, text)
        if phones:
            contact_info['phone'] = ''.join(phones[0])
        
        # LinkedIn pattern
        linkedin_pattern = r'linkedin\.com/in/([A-Za-z0-9-]+)'
        linkedin = re.search(linkedin_pattern, text)
        if linkedin:
            contact_info['linkedin'] = linkedin.group(0)
        
        # GitHub pattern
        github_pattern = r'github\.com/([A-Za-z0-9-]+)'
        github = re.search(github_pattern, text)
        if github:
            contact_info['github'] = github.group(0)
        
        return contact_info
    
    def extract_certifications(self, text: str) -> List[str]:
        """Extract certifications from resume"""
        cert_keywords = [
            'certified', 'certification', 'certificate', 'aws', 'azure', 'google cloud',
            'pmp', 'cissp', 'cisa', 'cism', 'comptia', 'cisco', 'microsoft',
            'oracle', 'salesforce', 'scrum master', 'agile', 'six sigma'
        ]
        
        certifications = []
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in cert_keywords):
                # Clean and limit length
                clean_line = line.strip()
                if 10 <= len(clean_line) <= 100:
                    certifications.append(clean_line)
        
        return certifications[:5]  # Return top 5
    
    def analyze_resume_structure(self, text: str) -> Dict[str, bool]:
        """Analyze resume structure and completeness"""
        structure = {
            'has_contact_info': bool(self.extract_contact_info(text)),
            'has_education': 'education' in text.lower() or 'degree' in text.lower(),
            'has_experience': 'experience' in text.lower() or 'work' in text.lower(),
            'has_skills': 'skills' in text.lower() or 'technologies' in text.lower(),
            'has_projects': 'project' in text.lower(),
            'has_certifications': bool(self.extract_certifications(text)),
            'word_count': len(text.split()),
            'is_comprehensive': len(text.split()) >= 100
        }
        
        return structure
    
    def get_resume_quality_score(self, text: str) -> float:
        """Calculate overall resume quality score"""
        structure = self.analyze_resume_structure(text)
        skills = self.extract_skills(text)
        
        score = 0.0
        
        # Structure components (40% of score)
        if structure['has_contact_info']:
            score += 0.1
        if structure['has_education']:
            score += 0.1
        if structure['has_experience']:
            score += 0.1
        if structure['has_skills']:
            score += 0.1
        
        # Content quality (40% of score)
        if structure['word_count'] >= 200:
            score += 0.2
        elif structure['word_count'] >= 100:
            score += 0.1
        
        if len(skills.get('all_skills', [])) >= 5:
            score += 0.2
        elif len(skills.get('all_skills', [])) >= 3:
            score += 0.1
        
        # Bonus features (20% of score)
        if structure['has_projects']:
            score += 0.1
        if structure['has_certifications']:
            score += 0.1
        
        return min(score, 1.0)