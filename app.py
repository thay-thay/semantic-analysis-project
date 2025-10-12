"""
Main Streamlit Application for Semantic Analysis Project

This module handles:
- UI/UX and page configuration
- Form submission and data collection
- GitHub CSV persistence
- Results visualization
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import base64
import json
import plotly.express as px

# Import the semantic engine
from semantic_engine import run_semantic_analysis, load_reference_data

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
FILE_PATH = "data/user_responses.csv"


def append_to_github_csv(new_response):
    """Append a new user response to the CSV file stored on GitHub.
    
    Args:
        new_response (dict): Dictionary containing all form data
                            
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


def display_results(results, first_name):
    """Display comprehensive analysis results with visualizations.
    
    Args:
        results (dict): Analysis results from semantic engine
        first_name (str): User's first name for personalization
    """
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
    
    # Numeric list to build both the table and the chart
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
            
        # Small bar chart for the top-3 inside the expander
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
            # Prepare response dictionary
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

            # Save to GitHub
            with st.spinner("Saving your responses to GitHub..."):
                success = append_to_github_csv(responses)

            if success:
                st.success(f"✅ Thank you {first_name}! Your responses have been submitted successfully.")
                
                # Run semantic analysis using the engine
                with st.spinner("🧠 Analyzing your profile with advanced semantic matching..."):
                    results, error = run_semantic_analysis(responses)
                
                if error:
                    st.error(f"❌ Analysis failed: {error}")
                    
                elif results:
                    # Display comprehensive results
                    display_results(results, first_name)

            else:
                st.error("❌ Failed to save responses to GitHub. Please try again or contact support.")

