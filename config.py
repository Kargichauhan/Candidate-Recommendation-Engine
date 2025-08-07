# Configuration file for the Candidate Recommendation Engine

# AI API Configuration
OPENAI_API_KEY = None  # Set this in Streamlit secrets or environment variable

# Model Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # You can change this to other models
MAX_CANDIDATES_DISPLAY = 10

# UI Configuration
APP_TITLE = "🎯 AI-Powered Candidate Recommendation Engine"
APP_DESCRIPTION = """
**Find the perfect candidates for your job openings using advanced AI matching!**

Upload resumes, paste a job description, and get intelligent recommendations with AI-powered explanations.
"""

# Similarity thresholds for color coding
SIMILARITY_THRESHOLDS = {
    'excellent': 0.8,
    'good': 0.7,
    'fair': 0.6
}

# Technical skills database
TECH_SKILLS_DB = {
    'programming': [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 
        'swift', 'kotlin', 'php', 'ruby', 'scala', 'r', 'matlab', 'sql'
    ],
    'web_frontend': [
        'react', 'angular', 'vue', 'html', 'css', 'sass', 'less', 'bootstrap',
        'tailwind', 'jquery', 'webpack', 'vite', 'nextjs', 'nuxt'
    ],
    'web_backend': [
        'node.js', 'express', 'django', 'flask', 'fastapi', 'spring', 'laravel',
        'rails', 'asp.net', 'graphql', 'rest', 'microservices'
    ],
    'data_science': [
        'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
        'matplotlib', 'seaborn', 'plotly', 'jupyter', 'spark', 'hadoop'
    ],
    'cloud_devops': [
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible',
        'jenkins', 'gitlab-ci', 'github-actions', 'prometheus', 'grafana'
    ],
    'databases': [
        'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
        'dynamodb', 'sqlite', 'oracle', 'sqlserver'
    ],
    'mobile': [
        'ios', 'android', 'react-native', 'flutter', 'xamarin', 'ionic'
    ],
    'ai_ml': [
        'machine-learning', 'deep-learning', 'nlp', 'computer-vision', 'llm',
        'transformers', 'bert', 'gpt', 'opencv', 'reinforcement-learning'
    ]
}