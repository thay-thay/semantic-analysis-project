"""
Semantic Analysis Engine for Competency Mapping and Job Recommendation

# -----------------------------------------------------------------------------
# SEMANTIC_ENGINE.PY — Semantic Job-Matching with SBERT
# -----------------------------------------------------------------------------
# PURPOSE:
#   Given a user's free-text responses to job/skill questions, this module 
#   computes semantic similarity between the user's profile and a library of 
#   predefined competencies. It then recommends jobs based on how well the 
#   user matches the required competencies for each job.
#
# ARCHITECTURE:
# - INPUT  : Form responses via Streamlit (dict format)
# - DATA   : data/competencies.csv, data/job_skills.csv, data/job_weights.csv,
#            data/questions.csv
# - OUTPUT : 
#     Competency scores (dict)          - each competency + similarity score
#     Block scores (dict)               - average score per competency block
#     Job recommendations (list)        - each job + final weighted score
#     Summary results (dict)            - compact summary for the front-end
#
# WORKFLOW (high-level):
# 1) Reads the user's free-text answers from the form submission
# 2) Uses SBERT (SentenceTransformer) to turn texts into vectors (embeddings)
# 3) Computes cosine similarity between user responses and every competency
# 4) Applies question weights to prioritize important questions
# 5) Aggregates competency scores using weighted averages
# 6) Computes block-level scores (average per competency category)
# 7) Matches against jobs using block-weighted scoring
# 8) Returns Top-K job recommendations with detailed metrics
#
# TECHNOLOGY STACK:
#   - sentence-transformers: SBERT model for semantic embeddings
#   - torch: Deep learning backend for embeddings
#   - pandas: Data manipulation and CSV handling
#   - numpy: Numerical computations
#   - streamlit: Caching mechanisms (@st.cache_resource, @st.cache_data)
#
# KEY COMPONENTS:
#   - Model: all-mpnet-base-v2 (768-dimensional embeddings)
#   - Similarity metric: Cosine similarity
#   - Score normalization: Maps raw scores (0.0-0.4) to user-friendly (0-100%)
#   - Question weighting: Technical questions weighted higher than opinion
#   - Block weighting: Job-specific importance for competency categories
#
# REQUIREMENTS (requirements.txt):
#   sentence-transformers>=2.2.0
#   torch>=2.0.0
#   pandas>=2.0.0
#   numpy>=1.24.0
#   streamlit>=1.28.0
# -----------------------------------------------------------------------------
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# --------------------------- Configuration (edit here) ------------------------

# Pretrained model used to compute sentence embeddings.
# "all-mpnet-base-v2" = strong, general-purpose, good quality.
# 768-dimensional embeddings, trained on 1B+ sentence pairs
MODEL_NAME: str = "all-MiniLM-L6-v2"

# Folder layout (relative paths so it works the same locally and in production)
DATA_DIR = Path("data")

# === Score Normalization Parameters ===
# Raw cosine similarity scores typically range from 0.0 to 0.4 for real user responses
# We normalize these to a 0-100% scale for better user interpretation
SCORE_MIN_THRESHOLD = 0.0  # Below this threshold → 0%
SCORE_MAX_EXPECTED = 0.35   # At or above this → 100%
SCORE_SCALING_FACTOR = 1.0  # Additional multiplier (1.0 = no extra scaling)

# Mapping des champs du formulaire vers les QuestionIDs
# This maps form field names to internal question identifiers
QUESTION_MAPPING = {
    "Programming": "Q01",           # Programming experience (text)
    "Data_Analysis": "Q02",         # Data analysis approach (text)
    "ML_Projects": "Q03",           # ML project experience (text)
    "ML_Problem": "Q04",            # ML problem-solving (text)
    "NLP": "Q05",                   # NLP experience (text)
    "Data_Pipeline": "Q06",         # Data pipeline experience (text)
    "Sharing_Results": "Q07",       # Communication/reporting (text)
    "Git_Level": "Q08",             # Git proficiency (Likert 1-5)
    "Presentation_Level": "Q09",    # Presentation skills (Likert 1-5)
    "Reflection": "Q10"             # DS/DE philosophy (text)
}

# ------------------------------ Helper functions ------------------------------

def normalize_score(raw_score: float, 
                    min_threshold: float = SCORE_MIN_THRESHOLD, 
                    max_expected: float = SCORE_MAX_EXPECTED, 
                    scaling_factor: float = SCORE_SCALING_FACTOR) -> float:
    """
    Normalize semantic similarity scores to a more user-friendly 0-1 scale.
    
    Raw cosine similarity scores from SBERT typically range from 0.0 to 0.4 for real
    user responses. This function maps those scores to a 0-100% range where:
    - Scores below min_threshold are mapped to 0%
    - Scores at or above max_expected are mapped to 100%
    - Scores in between are scaled linearly
    
    This makes the scores more intuitive for users. For example:
    - Raw score of 0.30 → 100% (mapped to max)
    - Raw score of 0.25 → 75% (linear interpolation)
    - Raw score of 0.15 → 25% (linear interpolation)
    - Raw score of 0.08 → 0% (below threshold)
    
    Arguments:
        raw_score (float): Original cosine similarity score (typically 0.0 to 0.4)
        min_threshold (float): Minimum score to consider meaningful (default 0.10)
        max_expected (float): Maximum realistic score, mapped to 1.0 (default 0.30)
        scaling_factor (float): Additional multiplier if needed (default 1.0)
    
    Returns:
        float: Normalized score between 0.0 and 1.0 (multiply by 100 for percentage)
    """
    if raw_score < min_threshold:
        return 0.0
    
    if raw_score >= max_expected:
        return 1.0
    
    # Linear scaling: (score - min) / (max - min)
    normalized = (raw_score - min_threshold) / (max_expected - min_threshold)
    normalized *= scaling_factor
    
    return min(1.0, max(0.0, normalized))


# -------------------------- Model & Data Loading ------------------------------

@st.cache_resource
def load_model() -> SentenceTransformer:
    """Load and cache the SBERT model for semantic analysis.
    
    This function uses Streamlit's cache_resource decorator to load the model
    only once and reuse it across multiple requests, improving performance.
    
    The model (all-mpnet-base-v2) is downloaded from Hugging Face on first run
    and cached locally. Subsequent runs reuse the cached model.
    
    Returns:
        SentenceTransformer: Pre-trained SBERT model for encoding text into 
                            768-dimensional embeddings
    """
    return SentenceTransformer(MODEL_NAME)


@st.cache_data
def load_reference_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and cache all reference CSV files needed for semantic analysis.
    
    This function loads four CSV files containing the reference data:
    - competencies.csv: Skills/competencies with their IDs, texts, and category blocks
    - job_skills.csv: Mapping between job positions and required competencies
    - job_weights.csv: Importance weights for each competency block per job
    - questions.csv: Survey questions with their types and competency mappings
    
    Data is cached using Streamlit's @st.cache_data to improve performance 
    and avoid reloading files on every request.
    
    File structure:
    - competencies.csv: CompetencyID, CompetencyText, BlockName
    - job_skills.csv: JobID, JobTitle, CompetencyID
    - job_weights.csv: JobID, BlockName, BlockWeight
    - questions.csv: QuestionID, QuestionText, Type, RelatedCompetencies
    
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
def precompute_competency_embeddings(_model: SentenceTransformer, 
                                     competencies_df: pd.DataFrame) -> Dict[str, Any]:
    """Precompute and cache vector embeddings for all competencies.
    
    This function generates semantic embeddings (vector representations) for
    all competency texts using the SBERT model. Embeddings are computed once
    and cached to avoid redundant computation.
    
    Each competency text is encoded into a 768-dimensional vector that captures
    its semantic meaning. These vectors are used later to compute similarity
    with user responses.
    
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
        # Encode text into 768-dimensional vector
        embedding = _model.encode(comp_text, convert_to_tensor=True)
        embeddings[comp_id] = embedding
    return embeddings


# ----------------------- Response Analysis Functions --------------------------

def analyze_single_response_semantic(question_id: str, 
                                     user_response: Any, 
                                     response_type: str, 
                                     model: SentenceTransformer, 
                                     competency_embeddings: Dict[str, Any]) -> Dict[str, float]:
    """
    Analyze a single user response and compute semantic similarity scores with all competencies.
    
    This function processes one answer from the user and calculates how relevant it is
    to each competency in the system. The scoring method depends on the response type:
    
    For TEXT responses:
    - Encodes the user's answer into a 768-dimensional embedding vector
    - Computes cosine similarity with each competency embedding
    - Returns raw similarity scores (0.0 to ~0.4 typically)
    
    For LIKERT responses (1-5 scale):
    - Normalizes the rating to 0.0-1.0 scale (divides by 5)
    - Scales down by 0.3 to reflect lower confidence vs text analysis
    - Applies the same score to all competencies (non-discriminative)
    
    Arguments:
        question_id (str): The ID of the question being answered (e.g., 'Q01')
        user_response (str or int): The user's answer text or numeric rating
        response_type (str): Type of response - either 'text' for open-ended or 'likert' for 1-5 scale
        model (SentenceTransformer): The SBERT model used to encode user text
        competency_embeddings (dict): Pre-computed embeddings for all competencies
                                     Format: {comp_id: embedding_tensor}
    
    Returns:
        dict: Dictionary mapping each CompetencyID to its similarity score
              Format: {comp_id: score}
              Example: {'C01': 0.25, 'C02': 0.18, 'C03': 0.31, ...}
    """
    
    # Handle Likert scale questions (1-5 ratings)
    if response_type == 'likert':
        try:
            # Normalize to 0-1 scale and apply dampening factor
            likert_score = float(user_response) / 5.0
            # Apply to all competencies with 0.3 multiplier
            return {comp_id: likert_score * 0.3 for comp_id in competency_embeddings.keys()}
        except:
            # If parsing fails, return zero scores
            return {comp_id: 0.0 for comp_id in competency_embeddings.keys()}
    
    # Handle text responses
    elif response_type == 'text':
        # Validate response has meaningful content
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
            # Ensure non-negative scores
            semantic_sim = max(0.0, semantic_sim)
            comp_scores[comp_id] = semantic_sim

        return comp_scores
    
    # Unknown response type
    return {}


def analyze_all_responses_weighted(user_responses: Dict[str, Any], 
                                   questions_df: pd.DataFrame, 
                                   model: SentenceTransformer, 
                                   competency_embeddings: Dict[str, Any], 
                                   question_weights: Dict[str, float] = None) -> Dict[str, float]:
    """
    Analyze all user responses with question weighting and compute aggregated competency scores.
    
    This function processes all answers from the user across all questions. It applies
    importance weights to different questions (e.g., technical ML questions count more
    than opinion questions) and computes a weighted average score for each competency.
    
    The weighting system reflects that:
    - Technical/practical questions are more important (weight 1.2-1.5)
    - Core ML/NLP experience is highly valued (weight 1.5)
    - Likert scale questions are less reliable (weight 0.7)
    - Opinion questions have moderate importance (weight 0.7-1.0)
    
    Algorithm:
    1. For each question-answer pair:
       a. Get the question's response type (text or likert)
       b. Get the question's importance weight
       c. Compute competency similarity scores for this response
       d. Multiply each score by the question weight
       e. Accumulate weighted scores and weights
    2. Compute weighted average for each competency:
       weighted_avg = sum(score × weight) / sum(weights)
    
    Arguments:
        user_responses (dict): Mapping of QuestionID to user's answer
                              Format: {'Q01': 'I use Python...', 'Q02': '...', ...}
        questions_df (DataFrame): DataFrame containing question metadata (QuestionID, Type, etc.)
        model (SentenceTransformer): SBERT model for encoding text responses
        competency_embeddings (dict): Pre-computed embeddings for all competencies
        question_weights (dict, optional): Custom weights per question. If None, uses default weights
    
    Returns:
        dict: Weighted average competency scores aggregated across all responses
              Format: {comp_id: weighted_avg_score}
              Example: {'C01': 0.23, 'C02': 0.18, ...}
    """
    
    if question_weights is None:
        # Default question importance weights
        question_weights = {
            'Q01': 1.0,    # Programming experience - Standard weight
            'Q02': 1.2,    # Data analysis approach - Important (core data skill)
            'Q03': 1.5,    # ML projects - Very important (practical experience)
            'Q04': 1.2,    # ML problem solving - Important (technical thinking)
            'Q05': 1.5,    # NLP experience - Very important (specialized skill)
            'Q06': 1.3,    # Data pipelines - Important (engineering skill)
            'Q07': 1.0,    # Communication/reporting - Standard (soft skill)
            'Q08': 0.7,    # Likert: Git skills - Less important (subjective rating)
            'Q09': 0.7,    # Likert: Presentation skills - Less important (subjective rating)
            'Q10': 0.7     # Opinion: What makes a good DS - Moderate (philosophical)
        }

    # Initialize accumulators for weighted scoring
    all_comp_scores = {}  # Accumulates: sum of (score × weight)
    comp_weights = {}     # Accumulates: sum of weights

    # Process each question-answer pair
    for question_id, answer in user_responses.items():
        # Look up question metadata
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
            
    # Compute weighted average for each competency
    # Formula: weighted_average = sum(score × weight) / sum(weights)
    for comp_id in all_comp_scores:
        if comp_weights[comp_id] > 0:
            all_comp_scores[comp_id] /= comp_weights[comp_id]
        else:
            all_comp_scores[comp_id] = 0.0

    return all_comp_scores


# ----------------------- Score Aggregation Functions --------------------------

def compute_block_scores(competency_scores: Dict[str, float], 
                        competencies_df: pd.DataFrame) -> Dict[str, float]:
    """Compute average scores for each competency block.
    
    Competencies are grouped into blocks (categories) such as:
    - Programming & Software Engineering
    - Machine Learning & AI
    - Data Engineering & Pipelines
    - Communication & Collaboration
    
    This function calculates the average score for each block based on individual 
    competency scores within that block.
    
    Algorithm:
    1. For each competency score:
       a. Look up which block it belongs to
       b. Add score to block's accumulator
       c. Increment block's count
    2. Compute average: block_score = sum(scores) / count
    
    Args:
        competency_scores (dict): Individual competency scores
                                 Format: {comp_id: score}
        competencies_df (pd.DataFrame): DataFrame with competency metadata including BlockName
        
    Returns:
        dict: Average score for each competency block
              Format: {block_name: average_score}
              Example: {'Programming': 0.25, 'ML & AI': 0.31, ...}
    """
    # Initialize accumulators  
    block_scores = {}
    block_counts = {}

    # Process each competency score
    for comp_id, score in competency_scores.items():
        # Look up block name for this competency
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


# ------------------------- Job Matching Functions -----------------------------

def compute_job_scores_weighted(competency_scores: Dict[str, float], 
                               job_skills_df: pd.DataFrame, 
                               job_weights_df: pd.DataFrame, 
                               competencies_df: pd.DataFrame) -> List[Tuple[str, str, float, Dict]]:
    """Calculate weighted match scores for all jobs based on user's competency profile.
    
    This function computes how well a user matches each job by:
    1. Looking at which competencies each job requires
    2. Applying job-specific block-level weights (some jobs value certain blocks more)
    3. Computing weighted average score and coverage metrics
    
    Block weights reflect job-specific priorities. For example:
    - Data Engineer: Higher weight on "Data Engineering" block
    - ML Researcher: Higher weight on "ML & AI" block
    - Data Analyst: Higher weight on "Data Analysis" block
    
    Algorithm for each job:
    1. Get list of required competencies
    2. Get block weights for this specific job
    3. For each required competency:
       a. Look up user's score for that competency
       b. Look up which block it belongs to
       c. Multiply score by block weight
       d. Accumulate weighted scores and weights
    4. Compute weighted average: sum(score × block_weight) / sum(block_weights)
    5. Normalize score to 0-100% scale
    6. Calculate coverage: % of competencies where user scored >= 0.22
    
    Args:
        competency_scores (dict): User's scores for each competency
        job_skills_df (pd.DataFrame): Mapping of jobs to required competencies
        job_weights_df (pd.DataFrame): Block weights for each job
        competencies_df (pd.DataFrame): Competency metadata with block assignments
        
    Returns:
        list: List of tuples sorted by match score (descending)
              Format: [(job_id, job_title, job_score, details_dict), ...]
              details_dict contains:
                - required_competencies: Total number required
                - covered_competencies: Number where user scored >= 0.22
                - coverage_percentage: % of competencies covered
                - competency_scores: Individual scores for each required competency
                - weighted_score: Final normalized score (0-1)
                - unweighted_score: Simple average of competency scores
                - block_weights_used: Block weights applied for this job
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

        # Normalize to 0-100% scale
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


# ----------------------- Main Recommendation Pipeline -------------------------

def recommend_jobs(user_responses: Dict[str, Any], 
                  competencies_df: pd.DataFrame, 
                  job_skills_df: pd.DataFrame, 
                  job_weights_df: pd.DataFrame, 
                  questions_df: pd.DataFrame, 
                  model: SentenceTransformer, 
                  competency_embeddings: Dict[str, Any], 
                  top_k: int = 3) -> Dict[str, Any]:
    """Complete pipeline for job recommendation based on user responses.
    
    This is the main orchestration function that coordinates the entire analysis:
    1. Analyzes all user responses to compute competency scores
    2. Computes block-level scores (category averages)
    3. Matches user profile against all jobs with weighted scoring
    4. Normalizes all scores at the END (to avoid breaking calculations)
    5. Returns top K recommendations with detailed metrics
    
    Pipeline stages:
    
    STAGE 1: Response Analysis
    - Processes each user answer (text or Likert)
    - Computes semantic similarity with all competencies
    - Applies question-level weights
    - Produces: raw_competency_scores dict (RAW, not normalized yet)
    
    STAGE 2: Block Aggregation
    - Groups competencies by category (blocks)
    - Computes average score per block
    - Produces: raw_block_scores dict (RAW, not normalized yet)
    
    STAGE 3: Job Matching
    - For each job, applies job-specific block weights
    - Computes weighted average match score
    - Calculates coverage metrics
    - Normalizes job scores inside the function
    - Produces: Ranked list of all jobs (with normalized scores)
    
    STAGE 4: Result Formatting & Normalization
    - Selects top K jobs
    - Normalizes competency and block scores for display
    - Packages results with full details
    - Returns comprehensive dictionary
    
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
            - 'competency_scores': Normalized individual competency scores (dict)
            - 'raw_competency_scores': Original competency scores before normalization (dict)
            - 'block_scores': Normalized average scores per block (dict)
            - 'raw_block_scores': Original block scores before normalization (dict)
            - 'job_recommendations': Top K jobs with full details (list)
            - 'all_jobs': All jobs ranked by match score (list)
    """
    
    # STAGE 1: Analyze all responses with weighted aggregation
    # IMPORTANT: Keep RAW scores here, don't normalize yet!
    raw_competency_scores = analyze_all_responses_weighted(
        user_responses, questions_df, model, competency_embeddings
    )
    
    # STAGE 2: Compute block-level scores using RAW competency scores
    # IMPORTANT: Use RAW scores to maintain calculation integrity
    raw_block_scores = compute_block_scores(raw_competency_scores, competencies_df)
    
    # STAGE 3: Compute job match scores with block weighting
    # This function uses RAW competency scores and normalizes internally
    all_job_scores = compute_job_scores_weighted(
        raw_competency_scores, job_skills_df, job_weights_df, competencies_df
    )
    
    # STAGE 4: Format results - get top K recommendations
    top_jobs = all_job_scores[:top_k]
    
    # === NORMALIZE SCORES FOR DISPLAY (at the very end) ===
    # Apply normalization ONLY now, after all calculations are complete
    # This ensures internal calculations use raw scores while users see friendly percentages
    
    # Normalize individual competency scores (0.20 → 60%, 0.30 → 85%, etc.)
    competency_scores_normalized = {
        comp_id: normalize_score(score) 
        for comp_id, score in raw_competency_scores.items()
    }
    
    # Normalize block scores (0.25 → 70%, 0.28 → 80%, etc.)
    block_scores_normalized = {
        block_name: normalize_score(score) 
        for block_name, score in raw_block_scores.items()
    }
    
    return {
        # NORMALIZED scores for user display (friendly percentages)
        'competency_scores': competency_scores_normalized,
        'block_scores': block_scores_normalized,
        
        # RAW scores for internal analysis and debugging
        'raw_competency_scores': raw_competency_scores,
        'raw_block_scores': raw_block_scores,
        
        # Job recommendations (already have normalized scores from stage 3)
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
# ----------------------- Main Entry Point -------------------------------------

def run_semantic_analysis(form_responses: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Execute the complete semantic analysis workflow on form responses.
    
    This is the main entry point for analyzing a user's form submission. It:
    1. Loads all necessary models and reference data (cached)
    2. Converts form responses to the internal QuestionID format
    3. Runs the complete job recommendation pipeline
    4. Computes overall coverage metrics
    5. Returns results or error message
    
    The function handles the full workflow from raw form data to final recommendations.
    It uses caching extensively to optimize performance on repeated calls.
    
    Error handling:
    - Returns (None, error_message) if any step fails
    - Returns (results_dict, None) on success
    
    Args:
        form_responses (dict): Raw form data from Streamlit
                              Format: {'First_Name': 'John', 'Programming': 'I use Python...', ...}
                              Must include all fields defined in QUESTION_MAPPING
                              
    Returns:
        tuple: (results_dict, error_message)
            - If successful: (results_dict, None)
            - If error: (None, error_string)
            
            results_dict contains:
                - 'competency_scores': Score for each competency (dict)
                - 'block_scores': Score for each competency block (dict)
                - 'job_recommendations': Top 5 job matches (list of dicts)
                - 'all_jobs': All jobs ranked by score (list of dicts)
                - 'final_coverage': Overall average coverage score (float)
                
    Example:
        >>> responses = {
        ...     'First_Name': 'Alice',
        ...     'Programming': 'I use Python and SQL daily',
        ...     'ML_Projects': 'Built a churn prediction model',
        ...     ...
        ... }
        >>> results, error = run_semantic_analysis(responses)
        >>> if error:
        ...     print(f"Error: {error}")
        ... else:
        ...     print(f"Top job: {results['job_recommendations'][0]['job_title']}")
    """
    try:
        # Load model and reference data (cached for performance)
        model = load_model()
        competencies, job_skills, job_weights, questions = load_reference_data()
        competency_embeddings = precompute_competency_embeddings(model, competencies)
        
        # Convert form field names to internal QuestionID format
        # Example: 'Programming' → 'Q01', 'Data_Analysis' → 'Q02'
        user_responses = {}
        for form_field, question_id in QUESTION_MAPPING.items():
            if form_field in form_responses:
                user_responses[question_id] = form_responses[form_field]
        
        # Execute the complete recommendation pipeline
        results = recommend_jobs(
            user_responses,
            competencies,
            job_skills,
            job_weights,
            questions,
            model,
            competency_embeddings,
            top_k=5  # Return top 5 job recommendations
        )
        
        # Calculate overall coverage score across all blocks
        # This gives a single metric for the user's overall competency level
        final_coverage = np.mean(list(results['block_scores'].values()))
        results['final_coverage'] = final_coverage
        
        return results, None
        
    except Exception as e:
        # Return error message if any step fails
        return None, str(e)
