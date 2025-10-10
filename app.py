import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import base64
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer, util

# importe ta page de visu si le module existe
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

@st.cache_resource
def load_model():
    """Cache le modèle SBERT"""
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_reference_data():
    """Cache les données de référence"""
    competencies = pd.read_csv(DATA_DIR / "competencies.csv")
    job_skills = pd.read_csv(DATA_DIR / "job_skills.csv")
    job_weights = pd.read_csv(DATA_DIR / "job_weights.csv")
    questions = pd.read_csv(DATA_DIR / "questions.csv")
    
    return competencies, job_skills, job_weights, questions

@st.cache_data
def precompute_competency_embeddings(_model, competencies_df):
    """Précompute les embeddings des compétences"""
    embeddings = {}
    for _, row in competencies_df.iterrows():
        comp_id = row['CompetencyID']
        comp_text = row['CompetencyText']
        embedding = _model.encode(comp_text, convert_to_tensor=True)
        embeddings[comp_id] = embedding
    return embeddings

def analyze_single_response_semantic(question_id, user_response, response_type, 
                                     model, competency_embeddings):
    """Analyse une réponse utilisateur (version du notebook)"""
    
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

        user_embedding = model.encode(str(user_response), convert_to_tensor=True)

        comp_scores = {}
        for comp_id, comp_embedding in competency_embeddings.items():
            semantic_sim = util.cos_sim(user_embedding, comp_embedding).item()
            semantic_sim = max(0.0, semantic_sim)
            comp_scores[comp_id] = semantic_sim

        return comp_scores
    
    return {}

def analyze_all_responses_weighted(user_responses, questions_df, model, 
                                   competency_embeddings, question_weights=None):
    """Analyse toutes les réponses avec pondération (version du notebook)"""
    
    if question_weights is None:
        question_weights = {
            'Q01': 1.0,
            'Q02': 1.2,
            'Q03': 1.5,
            'Q04': 1.2,
            'Q05': 1.5,
            'Q06': 1.3,
            'Q07': 1.0,
            'Q08': 0.5,
            'Q09': 0.5,
            'Q10': 0.7
        }
    
    all_comp_scores = {}
    comp_weights = {}

    for question_id, answer in user_responses.items():
        question_row = questions_df[questions_df['QuestionID'] == question_id]

        if question_row.empty:
            continue

        response_type = question_row.iloc[0]['Type']
        q_weight = question_weights.get(question_id, 1.0)

        comp_scores = analyze_single_response_semantic(
            question_id, answer, response_type, model, competency_embeddings
        )

        for comp_id, score in comp_scores.items():
            if comp_id not in all_comp_scores:
                all_comp_scores[comp_id] = 0.0
                comp_weights[comp_id] = 0.0
            
            all_comp_scores[comp_id] += score * q_weight
            comp_weights[comp_id] += q_weight
    
    # Compute weighted average
    for comp_id in all_comp_scores:
        if comp_weights[comp_id] > 0:
            all_comp_scores[comp_id] /= comp_weights[comp_id]
        else:
            all_comp_scores[comp_id] = 0.0

    return all_comp_scores

def compute_block_scores(competency_scores, competencies_df):
    """Calcule les scores par bloc (version du notebook)"""
    block_scores = {}
    block_counts = {}

    for comp_id, score in competency_scores.items():
        comp_row = competencies_df[competencies_df['CompetencyID'] == comp_id]

        if comp_row.empty:
            continue

        block_name = comp_row.iloc[0]['BlockName']
        if block_name not in block_scores:
            block_scores[block_name] = 0.0
            block_counts[block_name] = 0
        
        block_scores[block_name] += score
        block_counts[block_name] += 1
    
    for block in block_scores:
        if block_counts[block] > 0:
            block_scores[block] /= block_counts[block]
        else:
            block_scores[block] = 0.0
    
    return block_scores

def compute_job_scores_weighted(competency_scores, job_skills_df, 
                               job_weights_df, competencies_df):
    """Calcule les scores des jobs avec pondération (version du notebook)"""
    
    job_results = []

    for job_id in job_skills_df['JobID'].unique():
        job_rows = job_skills_df[job_skills_df['JobID'] == job_id]
        job_title = job_rows.iloc[0]['JobTitle']
        required_comps = job_rows['CompetencyID'].tolist()

        job_weight_rows = job_weights_df[job_weights_df['JobID'] == job_id]
        block_weights = dict(zip(job_weight_rows['BlockName'], job_weight_rows['BlockWeight']))

        weighted_score = []
        weights = []
        matched_scores = []

        for comp_id in required_comps:
            comp_row = competencies_df[competencies_df['CompetencyID'] == comp_id]
            if comp_row.empty:
                continue
            
            block_name = comp_row.iloc[0]['BlockName']
            block_weight = block_weights.get(block_name, 1.0)

            score = competency_scores.get(comp_id, 0.0)
            matched_scores.append(score)

            weighted_score.append(score * block_weight)
            weights.append(block_weight)
        
        if sum(weights) > 0:
            job_score = sum(weighted_score) / sum(weights)
        else:
            job_score = 0.0

        total_required = len(required_comps)
        covered_count = sum(1 for s in matched_scores if s >= 0.22)
        coverage_pct = (covered_count / total_required) * 100 if total_required > 0 else 0.0

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

    job_results.sort(key=lambda x: x[2], reverse=True)
    return job_results

def recommend_jobs(user_responses, competencies_df, job_skills_df, 
                  job_weights_df, questions_df, model, 
                  competency_embeddings, top_k=3):
    """Pipeline complet de recommandation (version du notebook)"""
    
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
    """Exécute l'analyse sémantique complète"""
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
                    
                    # Top 5 compétences
                    st.markdown("### 🏆 Top 5 Competencies")
                    competencies, _, _, _ = load_reference_data()
                    
                    top_5_comps = sorted(
                        results['competency_scores'].items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:5]
                    
                    comp_data = []
                    for comp_id, score in top_5_comps:
                        comp_text = competencies[competencies['CompetencyID'] == comp_id]['CompetencyText'].values[0]
                        block = competencies[competencies['CompetencyID'] == comp_id]['BlockName'].values[0]
                        comp_data.append({
                            'Competency': comp_text,
                            'Block': block,
                            'Score': f"{score:.1%}"
                        })
                    
                    st.dataframe(pd.DataFrame(comp_data), use_container_width=True)
                    
                    # Jobs recommandés
                    st.markdown("### 💼 Top 5 Recommended Jobs")
                    jobs_data = []
                    for job in results['job_recommendations']:
                        jobs_data.append({
                            'Rank': job['rank'],
                            'Job Title': job['job_title'],
                            'Match Score': f"{job['match_score']:.1%}",
                            'Coverage': f"{job['details']['covered_competencies']}/{job['details']['required_competencies']} ({job['details']['coverage_percentage']:.0f}%)"
                        })
                    
                    st.dataframe(pd.DataFrame(jobs_data), use_container_width=True)
                    
                    # Scores par bloc
                    st.markdown("### 📊 Competency Block Scores")
                    block_data = [
                        {'Block': block, 'Average Score': f"{score:.1%}"}
                        for block, score in sorted(results['block_scores'].items(), key=lambda x: x[1], reverse=True)
                    ]
                    st.dataframe(pd.DataFrame(block_data), use_container_width=True)
                    
                    # Détails du top job
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
            else:
                st.error("❌ Failed to save responses to GitHub. Please try again or contact support.")
