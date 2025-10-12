import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import base64
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer, util
import plotly.express as px
import plotly.graph_objects as go

# Import visualisation if exist
try:
    from viz_page import show_visualisations
    HAS_VIZ = True
except Exception:
    HAS_VIZ = False

# --- Navigation ---
pages = ["Accueil"]
if HAS_VIZ:
    pages.append("Visualisations")

choice = st.sidebar.radio("Navigation", pages, index=0)

# --- Si on choisit Visualisations : on affiche et on S'ARRÊTE ---
if HAS_VIZ and choice == "Visualisations":
    show_visualisations()
    st.stop()

# === Page configuration ===
st.set_page_config(
    page_title="Semantic Analysis Project",
    page_icon="🧠",
    layout="wide"
)

# === Custom CSS ===
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Roboto', sans-serif;
        }
        .header-title {
            font-size: 32px;
            font-weight: 700;
            color: #017179;
        }
        .header-subtitle {
            font-size: 18px;
            color: #017179;
            margin-bottom: 20px;
        }
        .stTextArea, .stSlider, .stTextInput {
            font-size: 16px;
        }
        .stButton>button {
            background-color: #017179;
            color: white;
            border-radius: 8px;
        }
        div[data-baseweb="slider"] > div > div > div > div > div[role="slider"] + div {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# === Display ECE Logo ===
logo_url = "https://raw.githubusercontent.com/thay-thay/semantic-analysis-project/main/data/ECE_LOGO_2021_web.png"
st.markdown(
    f"""
    <div style="display:flex; justify-content:center; margin-bottom:20px;">
        <img src="{logo_url}" width="200">
    </div>
    """,
    unsafe_allow_html=True
)

# === Header ===
st.markdown("""
<div>
    <div class="header-title">Project – Semantic Analysis</div>
    <div class="header-subtitle">Semantic Analysis for Competency Mapping and Job Profile Recommendation</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# === GitHub Configuration ===
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
GITHUB_REPO = "thay-thay/semantic-analysis-project"
FILE_PATH = "user_responses.csv"

# === Semantic Analysis Configuration ===
MODEL_NAME = "all-mpnet-base-v2"
DATA_DIR = Path("data")

# === Score Normalization Parameters ===
SCORE_MIN_THRESHOLD = 0.10  # Minimum score to consider (below this = 0%)
SCORE_MAX_EXPECTED = 0.30   # Maximum realistic score (this becomes 100%)
SCORE_SCALING_FACTOR = 1.0  # Additional multiplier if needed (1.0 = no extra scaling)

# Mapping des champs du formulaire vers les QuestionIDs
QUESTION_MAPPING = {
    "Programming": "Q01",
    "Data_Analysis": "Q02",
    "ML_Projects": "Q03",
    "ML_Problem": "Q04",
    "NLP": "Q05",
    "Data_Pipeline": "Q06",
    "Sharing_Results": "Q07",
    "Git_Level": "Q08",
    "Presentation_Level": "Q09",
    "Reflection": "Q10"
}

def normalize_score(raw_score, min_threshold=SCORE_MIN_THRESHOLD, 
                    max_expected=SCORE_MAX_EXPECTED, 
                    scaling_factor=SCORE_SCALING_FACTOR):
    """
    Normalize semantic similarity scores to a more user-friendly 0-1 scale.
    
    Raw cosine similarity scores from SBERT typically range from 0.0 to 0.4 for real
    user responses. This function maps those scores to a 0-100% range where:
    - Scores below min_threshold are mapped to 0%
    - Scores at or above max_expected are mapped to 100%
    - Scores in between are scaled linearly
    
    This makes the scores more intuitive for users. For example:
    - Raw score of 0.30 → 75% (instead of 30%)
    - Raw score of 0.25 → 50% (instead of 25%)
    - Raw score of 0.15 → 0% (minimum threshold)
    
    Arguments:
        raw_score (float): Original cosine similarity score (typically 0.0 to 0.4)
        min_threshold (float): Minimum score to consider meaningful (default 0.15)
        max_expected (float): Maximum realistic score, mapped to 1.0 (default 0.35)
        scaling_factor (float): Additional multiplier if needed (default 1.0)
    
    Returns:
        float: Normalized score between 0.0 and 1.0 (multiply by 100 for percentage)
    """
    # If score is below minimum threshold, return 0
    if raw_score < min_threshold:
        return 0.0
    
    # If score is at or above maximum expected, return 1.0 (capped at 100%)
    if raw_score >= max_expected:
        return 1.0
    
    # Linear scaling between min_threshold and max_expected
    # Formula: (score - min) / (max - min)
    normalized = (raw_score - min_threshold) / (max_expected - min_threshold)
    
    # Apply additional scaling factor if needed
    normalized *= scaling_factor
    
    # Ensure result is between 0.0 and 1.0
    return min(1.0, max(0.0, normalized))

@st.cache_resource
def load_model():
    """Load and cache the SBERT model for semantic analysis.
    
    This function uses Streamlit's cache_resource decorator to load the model
    only once and reuse it across multiple requests, improving performance.
    
    Returns:
        SentenceTransformer: Pre-trained SBERT model for encoding text into embeddings"""

    # Load the sentence transformer model from Hugging Face
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_reference_data():
    """Load and cache all reference CSV files needed for semantic analysis.
    
    This function loads four CSV files containing the reference data:
    - Competencies: Skills/competencies with their IDs, texts, and category blocks
    - Job Skills: Mapping between job positions and required competencies
    - Job Weights: Importance weights for each competency block per job
    - Questions: Survey questions with their types and competency mappings
    
    Data is cached to improve performance and avoid reloading files on every request.
    
    Arguments:
        None
    
    Returns:
        tuple: Four pandas DataFrames:
            - competencies (DataFrame): Competency reference data
            - job_skills (DataFrame): Job-to-competency mappings
            - job_weights (DataFrame): Block importance weights per job
            - questions (DataFrame): Question metadata and mappings
    """
    competencies = pd.read_csv(DATA_DIR / "competencies.csv")
    job_skills = pd.read_csv(DATA_DIR / "job_skills.csv")
    job_weights = pd.read_csv(DATA_DIR / "job_weights.csv")
    questions = pd.read_csv(DATA_DIR / "questions.csv")
    
    return competencies, job_skills, job_weights, questions

@st.cache_data
def precompute_competency_embeddings(_model, competencies_df):
    """Precompute and cache vector embeddings for all competencies.
    
    This function generates semantic embeddings (vector representations) for
    all competency texts using the SBERT model. Embeddings are computed once
    and cached.
    
    Arguments:
        _model (SentenceTransformer): The SBERT model used to encode text
                                     (underscore prefix tells Streamlit not to hash it)
        competencies_df (DataFrame): DataFrame containing CompetencyID and CompetencyText columns
    
    Returns:
        dict: Dictionary mapping CompetencyID to its embedding tensor
              Format: {comp_id: tensor([768 dimensions])}
              Example: {'C01': tensor([0.12, -0.45, ...]), 'C02': tensor([...])}
    """
    embeddings = {}
    for _, row in competencies_df.iterrows():
        comp_id = row['CompetencyID']
        comp_text = row['CompetencyText']
        embedding = _model.encode(comp_text, convert_to_tensor=True)
        embeddings[comp_id] = embedding
    return embeddings

def analyze_single_response_semantic(question_id, user_response, response_type, 
                                     model, competency_embeddings):
    """
    Analyze a single user response and compute semantic similarity scores with all competencies.
    
    This function processes one answer from the user and calculates how relevant it is
    to each competency in the system. For text responses, it uses cosine similarity
    between embeddings. For Likert scale responses (1-5 ratings), it applies a simple formula.
    
    Arguments:
        question_id (str): The ID of the question being answered  
        user_response (str or int): The user's answer text or numeric rating
        response_type (str): Type of response - either 'text' for open-ended or 'likert' for 1-5 scale
        model (SentenceTransformer): The SBERT model used to encode user text
        competency_embeddings (dict): Pre-computed embeddings for all competencies
                                     Format: {comp_id: embedding_tensor}
    
    Returns:
        dict: Dictionary mapping each CompetencyID to its similarity score
              Format: {comp_id: score}
    """
    
    # Handle Likert scale questions
    if response_type == 'likert':
        try:
            likert_score = float(user_response) / 5.0
            return {comp_id: likert_score * 0.3 for comp_id in competency_embeddings.keys()}
        except:
            return {comp_id: 0.0 for comp_id in competency_embeddings.keys()}
    
    # Handle text responses
    elif response_type == 'text':
        if not user_response or len(str(user_response).strip()) < 5:
            return {comp_id: 0.0 for comp_id in competency_embeddings.keys()}
            
        # Encode the user's text response into a vector embedding
        # This converts text into a 768-dimensional vector
        user_embedding = model.encode(str(user_response), convert_to_tensor=True)

        comp_scores = {}
        for comp_id, comp_embedding in competency_embeddings.items():
            # Calculate cosine similarity between the two vectors
            # Cosine similarity measures the angle between vectors (-1 to 1)
            # Higher values mean more semantic similarity
            semantic_sim = util.cos_sim(user_embedding, comp_embedding).item()
            semantic_sim = max(0.0, semantic_sim)
            comp_scores[comp_id] = semantic_sim

        return comp_scores
    
    return {}

def analyze_all_responses_weighted(user_responses, questions_df, model, 
                                   competency_embeddings, question_weights=None):
    """
    Analyze all user responses with question weighting and compute aggregated competency scores.
    
    This function processes all answers from the user across all questions. It applies
    importance weights to different questions (e.g., technical ML questions count more
    than opinion questions) and computes a weighted average score for each competency.
    
    The weighting system reflects that:
    - Technical questions are more important (weight 1.5)
    - Data analysis questions are important (weight 1.2)
    - Likert scale questions are less important (weight 0.7)
    - Opinion questions have moderate importance (weight 0.7-1.0)
    
    Arguments:
        user_responses (dict): Mapping of QuestionID to user's answer
        questions_df (DataFrame): DataFrame containing question metadata 
        model (SentenceTransformer): SBERT model for encoding text responses
        competency_embeddings (dict): Pre-computed embeddings for all competencies
        question_weights (dict, optional): Custom weights per question. If None, uses default weights
    
    Returns:
        dict: Weighted average competency scores aggregated across all responses
    """
    
    if question_weights is None:
        question_weights = {
            'Q01': 1.0,    # Programming experience - Standard
            'Q02': 1.2,    # Data analysis approach - Important (core skill)
            'Q03': 1.5,    # ML projects - Very important (practical experience)
            'Q04': 1.2,    # ML problem solving - Important
            'Q05': 1.5,    # NLP experience - Very important (specialized skill)    
            'Q06': 1.3,    # Data pipelines - Important (engineering skill)    
            'Q07': 1.0,    # Communication/reporting - Standard
            'Q08': 0.7,    # Likert: Git skills - Less important (subjective rating)
            'Q09': 0.7,    # Likert: Presentation skills - Less important (subjective rating)    
            'Q10': 0.7    # Opinion: What makes a good DS - Moderate (philosophical)
        }

    # Initialize accumulators for weighted scoring
    all_comp_scores = {} # sum of seighted score
    comp_weights = {} # sum of weights

    # Process each question-answer pair
    for question_id, answer in user_responses.items():
        question_row = questions_df[questions_df['QuestionID'] == question_id]

        if question_row.empty:
            continue

        # Extract response type ('text' or 'likert') from metadata
        response_type = question_row.iloc[0]['Type']

        # Get the importance weight for this question (default to 1.0 if not specified)
        q_weight = question_weights.get(question_id, 1.0)

        # Analyze this single response to get competency similarity scores
        comp_scores = analyze_single_response_semantic(
            question_id, answer, response_type, model, competency_embeddings
        )

        # Add weighted scores to the accumulators
        for comp_id, score in comp_scores.items():
            if comp_id not in all_comp_scores:
                all_comp_scores[comp_id] = 0.0
                comp_weights[comp_id] = 0.0
                
            # Add: (score × question_weight) to total
            all_comp_scores[comp_id] += score * q_weight

            # Add: question_weight to total weights (for computing average later)
            comp_weights[comp_id] += q_weight
            
   
    # Compute weighted average
    # Formula: weighted_average = sum(score × weight) / sum(weights)
    for comp_id in all_comp_scores:
        if comp_weights[comp_id] > 0:
            all_comp_scores[comp_id] /= comp_weights[comp_id]
        else:
            all_comp_scores[comp_id] = 0.0

    return all_comp_scores

def compute_block_scores(competency_scores, competencies_df):
    """Compute average scores for each competency block.
    
    Competencies are grouped into blocks.
    This function calculates the average score for each block based on individual 
    competency scores within that block.
    
    Args:
        competency_scores (dict): Individual competency scores
                                 Format: {comp_id: score}
        competencies_df (pd.DataFrame): DataFrame with competency metadata including BlockName
        
    Returns:
        dict: Average score for each competency block
              Format: {block_name: average_score}
    """
    # Initialize accumulators  
    block_scores = {}
    block_counts = {}

    # Process each competency score
    for comp_id, score in competency_scores.items():
        comp_row = competencies_df[competencies_df['CompetencyID'] == comp_id]

        if comp_row.empty:
            continue

        block_name = comp_row.iloc[0]['BlockName']
        if block_name not in block_scores:
            block_scores[block_name] = 0.0
            block_counts[block_name] = 0

        # Add score and increment count for this block
        block_scores[block_name] += score
        block_counts[block_name] += 1

    # Calculate average score for each block
    for block in block_scores:
        if block_counts[block] > 0:
            block_scores[block] /= block_counts[block]
        else:
            block_scores[block] = 0.0
    
    return block_scores

def compute_job_scores_weighted(competency_scores, job_skills_df, 
                               job_weights_df, competencies_df):
    """Calculate weighted match scores for all jobs based on user's competency profile.
    
    This function computes how well a user matches each job by:
    1. Looking at which competencies each job requires
    2. Applying block-level weights 
    3. Computing weighted average score and coverage metrics
    
    Args:
        competency_scores (dict): User's scores for each competency
        job_skills_df (pd.DataFrame): Mapping of jobs to required competencies
        job_weights_df (pd.DataFrame): Block weights for each job
        competencies_df (pd.DataFrame): Competency metadata with block assignments
        
    Returns:
        list: List of tuples sorted by match score (descending)
              Format: [(job_id, job_title, job_score, details_dict), ...]
              details_dict contains: required/covered competencies, coverage %, 
                                    competency scores, weighted/unweighted scores
    """
    
    job_results = []

    # Process each unique job
    for job_id in job_skills_df['JobID'].unique():
        # Get all rows for this job (one row per required competency)
        job_rows = job_skills_df[job_skills_df['JobID'] == job_id]
        job_title = job_rows.iloc[0]['JobTitle']
        required_comps = job_rows['CompetencyID'].tolist()

        # Get block weights for this specific job
        job_weight_rows = job_weights_df[job_weights_df['JobID'] == job_id]
        block_weights = dict(zip(job_weight_rows['BlockName'], job_weight_rows['BlockWeight']))

        # Initialize accumulators for scoring
        weighted_score = []    # Scores multiplied by block weights
        weights = []            # Block weights
        matched_scores = []    # Raw scores for coverage calculation

        # Score each required competency
        for comp_id in required_comps:
            # Look up which block this competency belongs to
            comp_row = competencies_df[competencies_df['CompetencyID'] == comp_id]
            if comp_row.empty:
                continue

            # Get block name and its weight for this job
            block_name = comp_row.iloc[0]['BlockName']
            block_weight = block_weights.get(block_name, 1.0)

            # Get user's score for this competency
            score = competency_scores.get(comp_id, 0.0)
            matched_scores.append(score)

            # Apply block weight to the score
            weighted_score.append(score * block_weight)
            weights.append(block_weight)

        # Calculate final weighted job score
        if sum(weights) > 0:
            raw_job_score = sum(weighted_score) / sum(weights)
        else:
            raw_job_score = 0.0

        job_score = normalize_score(raw_job_score)
        # Calculate coverage metrics
        total_required = len(required_comps)
        # Count how many competencies are "covered" (score >= 0.22 threshold)
        covered_count = sum(1 for s in matched_scores if s >= 0.22)
        coverage_pct = (covered_count / total_required) * 100 if total_required > 0 else 0.0

        # Compile detailed metrics for this job
        details = {
            'required_competencies': total_required,
            'covered_competencies': covered_count,
            'coverage_percentage': coverage_pct,
            'competency_scores': dict(zip(required_comps, matched_scores)),
            'weighted_score': job_score,
            'unweighted_score': np.mean(matched_scores) if matched_scores else 0.0,
            'block_weights_used': block_weights
        }

        job_results.append((job_id, job_title, job_score, details))
        
    # Sort jobs by weighted score (highest first)
    job_results.sort(key=lambda x: x[2], reverse=True)
    return job_results

def recommend_jobs(user_responses, competencies_df, job_skills_df, 
                  job_weights_df, questions_df, model, 
                  competency_embeddings, top_k=3):
    """Complete pipeline for job recommendation based on user responses.
    
    This is the main orchestration function that:
    1. Analyzes all user responses to compute competency scores
    2. Computes block-level scores
    3. Matches user profile against all jobs with weighted scoring
    4. Returns top K recommendations with detailed metrics
    
    Args:
        user_responses (dict): User's answers to all questions
                              Format: {'Q01': 'answer', 'Q02': 'answer', ...}
        competencies_df (pd.DataFrame): Competency reference data
        job_skills_df (pd.DataFrame): Job-competency mappings
        job_weights_df (pd.DataFrame): Block weights per job
        questions_df (pd.DataFrame): Question metadata
        model (SentenceTransformer): SBERT model for encoding
        competency_embeddings (dict): Pre-computed competency embeddings
        top_k (int, optional): Number of top jobs to return. Defaults to 3.
        
    Returns:
        dict: Comprehensive results dictionary containing:
            - 'competency_scores': Individual competency scores
            - 'block_scores': Average scores per competency block
            - 'job_recommendations': Top K jobs with full details
            - 'all_jobs': All jobs ranked by match score
    """
    
    # Step 1: Analyze responses
    competency_scores = analyze_all_responses_weighted(
        user_responses, questions_df, model, competency_embeddings
    )

    # Step 2: Compute block scores
    block_scores = compute_block_scores(competency_scores, competencies_df)

    # Step 3: Compute job scores
    all_job_scores = compute_job_scores_weighted(
        competency_scores, job_skills_df, job_weights_df, competencies_df
    )

    # Step 4: Get top K recommendations
    top_jobs = all_job_scores[:top_k]

    return {
        'competency_scores': competency_scores,
        'block_scores': block_scores,
        'job_recommendations': [
            {
                'rank': i+1,
                'job_id': job_id,
                'job_title': job_title,
                'match_score': score,
                'details': details
            }
            for i, (job_id, job_title, score, details) in enumerate(top_jobs)
        ],
        'all_jobs': [
            {
                'job_id': job_id,
                'job_title': job_title,
                'match_score': score,
                'details': details
            }
            for job_id, job_title, score, details in all_job_scores
        ]
    }

def run_semantic_analysis(form_responses):
    """Execute the complete semantic analysis workflow on form responses.
    
    This is the main entry point for analyzing a user's form submission. It:
    1. Loads all necessary models and reference data
    2. Converts form responses to the internal format
    3. Runs the job recommendation pipeline
    4. Computes overall coverage metrics
    
    Args:
        form_responses (dict): Raw form data from Streamlit
                              Format: {'First_Name': 'John', 'Programming': 'I use Python...', ...}
                              
    Returns:
        tuple: (results_dict, error_message)
            - If successful: (results_dict, None)
            - If error: (None, error_string)
            
            results_dict contains:
                - 'competency_scores': Score for each competency
                - 'block_scores': Score for each competency block
                - 'job_recommendations': Top 5 job matches
                - 'all_jobs': All jobs ranked
                - 'final_coverage': Overall average coverage score
    """
    try:
        # Charger le modèle et les données
        model = load_model()
        competencies, job_skills, job_weights, questions = load_reference_data()
        competency_embeddings = precompute_competency_embeddings(model, competencies)
        
        # Convertir les réponses du formulaire en format QuestionID
        user_responses = {}
        for form_field, question_id in QUESTION_MAPPING.items():
            if form_field in form_responses:
                user_responses[question_id] = form_responses[form_field]
        
        # Exécuter la recommandation
        results = recommend_jobs(
            user_responses,
            competencies,
            job_skills,
            job_weights,
            questions,
            model,
            competency_embeddings,
            top_k=5
        )
        
        # Calculer la couverture globale
        final_coverage = np.mean(list(results['block_scores'].values()))
        results['final_coverage'] = final_coverage
        
        return results, None
        
    except Exception as e:
        return None, str(e)

def append_to_github_csv(new_response):
    """Append a new user response to the CSV file stored on GitHub.
    
    This function handles saving form responses to a GitHub repository by:
    1. Fetching the existing CSV file from GitHub
    2. Appending the new response
    3. Committing the updated file back to GitHub
    
    Args:
        new_response (dict): Dictionary containing all form data
                            Format: {'Timestamp': '...', 'First_Name': '...', 'Programming': '...', ...}
                            
    Returns:
        bool: True if successfully saved, False otherwise
    """
    
    if not GITHUB_TOKEN:
        st.error("❌ GitHub token not configured. Please add it to Streamlit secrets.")
        return False

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{FILE_PATH}"

    try:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            file_data = response.json()
            content = base64.b64decode(file_data['content']).decode('utf-8')
            sha = file_data['sha']

            from io import StringIO
            existing_df = pd.read_csv(StringIO(content))
            new_df = pd.concat([existing_df, pd.DataFrame([new_response])], ignore_index=True)
        elif response.status_code == 404:
            new_df = pd.DataFrame([new_response])
            sha = None
        else:
            st.error(f"❌ Error fetching file: {response.status_code}")
            return False

        csv_content = new_df.to_csv(index=False)
        encoded_content = base64.b64encode(csv_content.encode()).decode()

        commit_data = {
            "message": f"Add response from {new_response['First_Name']} {new_response['Last_Name']}",
            "content": encoded_content,
            "branch": "main"
        }

        if sha:
            commit_data["sha"] = sha

        update_response = requests.put(url, headers=headers, data=json.dumps(commit_data))

        if update_response.status_code in [200, 201]:
            return True
        else:
            st.error(f"❌ Error updating file: {update_response.status_code}")
            st.error(update_response.json())
            return False

    except Exception as e:
        st.error(f"❌ Exception occurred: {str(e)}")
        return False


# === Form ===
with st.form("skills_form"):
    first_name = st.text_input("First Name", placeholder="Enter your first name")
    last_name = st.text_input("Last Name", placeholder="Enter your last name")

    prog_text = st.text_area(
        "Describe your experience with programming.",
        placeholder="Ex: I mostly use Python and SQL, and I work with Git and OOP concepts."
    )
    data_text = st.text_area(
        "Explain how you typically analyze a dataset.",
        placeholder="Ex: I clean the data, perform EDA, visualize distributions, and calculate statistics."
    )
    ml_text = st.text_area(
        "Tell us about a project where you applied machine learning.",
        placeholder="Ex: I built a regression model using scikit-learn and evaluated it with cross-validation."
    )
    ml_problem_text = st.text_area(
        "How would you approach designing a churn prediction model?",
        placeholder="Ex: I would perform feature engineering, select a model, train, and evaluate it."
    )
    nlp_text = st.text_area(
        "Have you ever worked with NLP?",
        placeholder="Ex: I tokenized text, used embeddings, transformers, sentiment analysis, and NER."
    )
    pipeline_text = st.text_area(
        "Explain a time when you built or maintained a data pipeline.",
        placeholder="Ex: I implemented an ETL pipeline using Airflow for batch processing."
    )
    sharing_text = st.text_area(
        "How do you usually share the results of your analysis?",
        placeholder="Ex: I create dashboards, visualizations, and prepare presentations to explain insights."
    )
    reflection_text = st.text_area(
        "What makes someone a strong Data Scientist / Engineer?",
        placeholder="Ex: Strong problem-solving, communication skills, and mastery of tools."
    )

    col1, col2 = st.columns(2)
    with col1:
        git_level = st.slider(
            "Git & Collaboration",
            min_value=1, max_value=5, value=3,
            help="1 = Beginner / Weak, 5 = Expert / Strong"
        )
    with col2:
        presentation_level = st.slider(
            "Presentation Skills",
            min_value=1, max_value=5, value=3,
            help="1 = Beginner / Weak, 5 = Expert / Strong"
        )

    submitted = st.form_submit_button("Submit")

    if submitted:
        if not first_name.strip() or not last_name.strip():
            st.warning("⚠️ Please fill in your First Name and Last Name before submitting.")
        else:
            responses = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "First_Name": first_name,
                "Last_Name": last_name,
                "Programming": prog_text,
                "Data_Analysis": data_text,
                "ML_Projects": ml_text,
                "ML_Problem": ml_problem_text,
                "NLP": nlp_text,
                "Data_Pipeline": pipeline_text,
                "Sharing_Results": sharing_text,
                "Git_Level": git_level,
                "Presentation_Level": presentation_level,
                "Reflection": reflection_text
            }

            with st.spinner("Saving your responses to GitHub..."):
                success = append_to_github_csv(responses)

            if success:
                st.success(f"✅ Thank you {first_name}! Your responses have been submitted successfully.")
                
                # Analyse sémantique avec le nouveau moteur
                with st.spinner("🧠 Analyzing your profile with advanced semantic matching..."):
                    results, error = run_semantic_analysis(responses)
                
                if error:
                    st.error(f"❌ Analysis failed: {error}")
                    
                elif results:
                    st.balloons()
                    st.success("✅ Analysis complete!")
                    
                    # Afficher les résultats
                    st.markdown("---")
                    st.markdown("## 🎯 Your Results")
                    
                    # Metrics principales
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Coverage", f"{results['final_coverage']:.1%}")
                    with col2:
                        top_job = results['job_recommendations'][0]
                        st.metric("Top Job Match", top_job['job_title'])
                    with col3:
                        st.metric("Match Score", f"{top_job['match_score']:.1%}")
                    
                    # TOP 5 COMPETENCIES
                    st.markdown("### 🏆 Top 5 Competencies")
                    competencies, _, _, _ = load_reference_data()
                    
                    # numeric list to build both the table and the chart
                    top_5_comps = sorted(
                        results['competency_scores'].items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:5]
                    
                    # Table (formatted)
                    comp_table_rows = []
                    for comp_id, score in top_5_comps:
                        comp_text = competencies[competencies['CompetencyID'] == comp_id]['CompetencyText'].values[0]
                        block = competencies[competencies['CompetencyID'] == comp_id]['BlockName'].values[0]
                        comp_table_rows.append({
                            'Competency': comp_text,
                            'Block': block,
                            'Score': f"{score:.1%}"
                        })
                    comp_table_df = pd.DataFrame(comp_table_rows)
                    st.dataframe(comp_table_df, use_container_width=True)
                    
                    # Chart (numeric)
                    comp_chart_df = pd.DataFrame([{
                        "CompetencyText": competencies.loc[competencies["CompetencyID"] == comp_id, "CompetencyText"].values[0],
                        "Score": float(score)
                    } for comp_id, score in top_5_comps])
                    if not comp_chart_df.empty:
                        fig_top = px.bar(
                            comp_chart_df.sort_values("Score"),
                            x="Score",
                            y="CompetencyText",
                            orientation="h",
                            range_x=[0, 1],
                            title=None,
                            color="Score",
                            color_continuous_scale="Blues"
                        )
                        fig_top.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(fig_top, use_container_width=True)
                    else:
                        st.info("No competency scores available for plotting.")
                    
                    # TOP 5 RECOMMENDED JOBS
                    st.markdown("### 💼 Top 5 Recommended Jobs")
                    # Table (formatted)
                    jobs_table_rows = []
                    for job in results['job_recommendations'][:5]:
                        jobs_table_rows.append({
                            'Rank': job['rank'],
                            'Job Title': job['job_title'],
                            'Match Score': f"{job['match_score']:.1%}",
                            'Coverage': f"{job['details']['covered_competencies']}/{job['details']['required_competencies']} ({job['details']['coverage_percentage']:.0f}%)"
                        })
                    jobs_table_df = pd.DataFrame(jobs_table_rows)
                    st.dataframe(jobs_table_df, use_container_width=True)
                    
                    # Chart (numeric)
                    jobs_chart_df = pd.DataFrame([{
                        "JobTitle": job['job_title'],
                        "Score": float(job['match_score'])
                    } for job in results['job_recommendations'][:5]])
                    if not jobs_chart_df.empty:
                        jobs_chart_df["Score"] = jobs_chart_df["Score"].clip(0, 1)
                        fig_jobs = px.bar(
                            jobs_chart_df.sort_values("Score"),
                            x="Score",
                            y="JobTitle",
                            orientation="h",
                            range_x=[0, 1],
                            title=None,
                            color="Score",
                            color_continuous_scale="Greens"
                        )
                        fig_jobs.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(fig_jobs, use_container_width=True)
                    else:
                        st.info("No job recommendations available for plotting.")
                    
                    # BLOCK SCORES
                    st.markdown("### 📊 Competency Block Scores")
                    # Table (formatted)
                    block_pairs_sorted = sorted(results['block_scores'].items(), key=lambda x: x[1], reverse=True)
                    block_table_rows = [{'Block': block, 'Average Score': f"{score:.1%}"} for block, score in block_pairs_sorted]
                    block_table_df = pd.DataFrame(block_table_rows)
                    st.dataframe(block_table_df, use_container_width=True)
                    
                    # Chart (numeric radar)
                    block_chart_df = pd.DataFrame([{"BlockName": block, "Score": float(score)} for block, score in block_pairs_sorted])
                    if {"BlockName", "Score"}.issubset(block_chart_df.columns) and len(block_chart_df) >= 3:
                        b = block_chart_df.copy()
                        b["Score"] = b["Score"].clip(0, 1)
                        fig_rad = px.line_polar(
                            b, r="Score", theta="BlockName", line_close=True, range_r=[0, 1], title=None
                        )
                        fig_rad.update_traces(fill="toself", line_color="#1f77b4")
                        fig_rad.update_layout(showlegend=False)
                        st.plotly_chart(fig_rad, use_container_width=True)
                    else:
                        st.info("Not enough blocks to draw a radar (need at least 3) or missing columns.")
                    
                    # TOP JOB — DETAILS
                    with st.expander("🔍 View detailed analysis for top match"):
                        top_job = results['job_recommendations'][0]
                        st.markdown(f"**{top_job['job_title']}**")
                        st.write(f"- Match Score: {top_job['match_score']:.1%}")
                        st.write(f"- Unweighted Score: {top_job['details']['unweighted_score']:.1%}")
                        st.write(f"- Coverage: {top_job['details']['coverage_percentage']:.1f}%")
                        
                        st.markdown("**Top matching competencies:**")
                        comp_scores = top_job['details']['competency_scores']
                        top_3 = sorted(comp_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                        
                        for comp_id, score in top_3:
                            comp_text = competencies[competencies['CompetencyID'] == comp_id]['CompetencyText'].values[0]
                            st.write(f"- {comp_text}: {score:.1%}")
                            
                        # small bar chart for the top-3 inside the expander
                        if top_3:
                            exp_df = pd.DataFrame([{
                                "CompetencyText": competencies.loc[competencies["CompetencyID"] == comp_id, "CompetencyText"].values[0],
                                "Score": float(score)
                            } for comp_id, score in top_3])
                            fig_top3 = px.bar(
                                exp_df.sort_values("Score"),
                                x="Score",
                                y="CompetencyText",
                                orientation="h",
                                range_x=[0, 1],
                                title=None,
                                color="Score",
                                color_continuous_scale="Blues"
                            )
                            fig_top3.update_layout(coloraxis_showscale=False)
                            st.plotly_chart(fig_top3, use_container_width=True)

            else:
                st.error("❌ Failed to save responses to GitHub. Please try again or contact support.")



