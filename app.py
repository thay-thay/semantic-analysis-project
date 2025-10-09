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
MODE = "avg"
TOP_K = 3
DATA_DIR = Path("data")
OUT_DIR = Path("outputs")

CANONICAL_TEXT_FIELDS = [
    "Programming", "Data_Analysis", "ML_Projects", "ML_Problem",
    "NLP", "Data_Pipeline", "Sharing_Results", "Reflection",
]

@st.cache_resource
def load_model():
    """Cache le modèle SBERT pour éviter de le recharger à chaque fois"""
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_reference_data():
    """Cache les données de référence (compétences et jobs)"""
    comp_path = DATA_DIR / "competencies.csv"
    jobs_path = DATA_DIR / "job_skills.csv"
    
    competencies = pd.read_csv(comp_path)
    jobs_long = pd.read_csv(jobs_path)
    
    competencies.columns = competencies.columns.str.strip()
    jobs_long.columns = jobs_long.columns.str.strip()
    
    jobs = (
        jobs_long
        .groupby(["JobID", "JobTitle"])["CompetencyID"]
        .apply(list)
        .reset_index()
        .rename(columns={"CompetencyID": "RequiredCompetencies"})
    )
    
    return competencies, jobs

def run_semantic_analysis(user_response_dict):
    """
    Exécute l'analyse sémantique pour une seule réponse utilisateur
    """
    try:
        # Charger le modèle et les données
        model = load_model()
        competencies, jobs = load_reference_data()
        
        # Construire le texte de l'utilisateur
        user_text = " ".join([
            str(user_response_dict.get(field, ""))
            for field in CANONICAL_TEXT_FIELDS
            if pd.notna(user_response_dict.get(field)) and str(user_response_dict.get(field)).strip()
        ])
        
        if not user_text.strip():
            return None, "No text to analyze"
        
        # Préparer les données de référence
        cid2block = dict(zip(competencies["CompetencyID"], competencies["BlockName"]))
        comp_ids = competencies["CompetencyID"].tolist()
        comp_texts = competencies["CompetencyText"].astype(str).tolist()
        
        # Encoder
        user_emb = model.encode([user_text], convert_to_tensor=True)
        comp_emb = model.encode(comp_texts, convert_to_tensor=True)
        
        # Calculer les scores de compétences
        if MODE == "avg":
            S = util.cos_sim(user_emb, comp_emb)
            comp_scores = S.squeeze(0).cpu().numpy()
        else:
            S = util.cos_sim(user_emb, comp_emb)
            comp_scores = S.max(dim=0).values.cpu().numpy()
        
        # Table des compétences
        comp_df = pd.DataFrame({
            "CompetencyID": comp_ids,
            "CompetencyText": comp_texts,
            "BlockName": [cid2block[c] for c in comp_ids],
            "Score": comp_scores
        }).sort_values("Score", ascending=False).reset_index(drop=True)
        
        # Scores par bloc
        block_scores = comp_df.groupby("BlockName")["Score"].mean().sort_values(ascending=False)
        final_coverage = float(block_scores.mean())
        
        # Scoring des jobs
        score_map = dict(zip(comp_df["CompetencyID"], comp_df["Score"]))
        
        def score_job_topk(required_ids, k=TOP_K):
            vals = [score_map.get(cid, 0.0) for cid in required_ids if cid in score_map]
            if not vals:
                return 0.0
            vals.sort(reverse=True)
            return float(np.mean(vals[:k]))
        
        jobs_copy = jobs.copy()
        jobs_copy["JobScore"] = jobs_copy["RequiredCompetencies"].apply(score_job_topk)
        jobs_ranked = jobs_copy.sort_values("JobScore", ascending=False).reset_index(drop=True)
        
        # Préparer les résultats
        top_job = jobs_ranked.iloc[0] if len(jobs_ranked) else None
        
        results = {
            "final_coverage": final_coverage,
            "top_job": top_job["JobTitle"] if top_job is not None else None,
            "top_job_score": float(top_job["JobScore"]) if top_job is not None else None,
            "top_competencies": comp_df.head(5)[["CompetencyID", "CompetencyText", "Score"]].to_dict(orient="records"),
            "all_jobs": jobs_ranked.head(10)[["JobTitle", "JobScore"]].to_dict(orient="records"),
            "block_scores": block_scores.to_dict()
        }
        
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
                
                # 🔥 NOUVELLE PARTIE : Analyse sémantique
                with st.spinner("🧠 Analyzing your profile and matching with job positions..."):
                    results, error = run_semantic_analysis(responses)
                
                if error:
                    st.error(f"❌ Analysis failed: {error}")
                elif results:
                    st.balloons()
                    st.success("✅ Analysis complete!")
                    
                    # Afficher les résultats
                    st.markdown("---")
                    st.markdown("## 🎯 Your Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Overall Coverage", f"{results['final_coverage']:.1%}")
                    with col2:
                        st.metric("Top Job Match", results['top_job'], f"{results['top_job_score']:.1%}")
                    
                    st.markdown("### 🏆 Top 5 Competencies")
                    comp_df = pd.DataFrame(results['top_competencies'])
                    comp_df['Score'] = comp_df['Score'].apply(lambda x: f"{x:.1%}")
                    st.dataframe(comp_df, use_container_width=True)
                    
                    st.markdown("### 💼 Recommended Jobs")
                    jobs_df = pd.DataFrame(results['all_jobs'])
                    jobs_df['JobScore'] = jobs_df['JobScore'].apply(lambda x: f"{x:.1%}")
                    st.dataframe(jobs_df, use_container_width=True)
                    
                    st.markdown("### 📊 Competency Block Scores")
                    block_df = pd.DataFrame(list(results['block_scores'].items()), columns=['Block', 'Score'])
                    block_df['Score'] = block_df['Score'].apply(lambda x: f"{x:.1%}")
                    st.dataframe(block_df, use_container_width=True)
            else:
                st.error("❌ Failed to save responses to GitHub. Please try again or contact support.")
