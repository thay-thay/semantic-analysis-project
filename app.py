import streamlit as st
import pandas as pd
from datetime import datetime
import os
from git import Repo

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

# === Header ===
st.markdown("""
<div>
    <div class="header-title">Project – Semantic Analysis</div>
    <div class="header-subtitle">Semantic Analysis for Competency Mapping and Job Profile Recommendation</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# === Form Section ===
with st.form("skills_form"):
    first_name = st.text_input("First Name", placeholder="Enter your first name")
    last_name = st.text_input("Last Name", placeholder="Enter your last name")

    prog_text = st.text_area("Describe your experience with programming.")
    data_text = st.text_area("Explain how you typically analyze a dataset.")
    ml_text = st.text_area("Tell us about a project where you applied ML techniques.")
    ml_problem_text = st.text_area("How would you design a model for predicting customer churn?")
    nlp_text = st.text_area("Have you ever worked with NLP techniques?")
    pipeline_text = st.text_area("Describe a time when you built a data pipeline.")
    sharing_text = st.text_area("How do you share results of your analysis?")
    col1, col2 = st.columns(2)
    with col1:
        git_level = st.slider("Git & Collaboration", 1, 5, 3)
    with col2:
        presentation_level = st.slider("Presentation Skills", 1, 5, 3)
    reflection_text = st.text_area("In your opinion, what makes a strong Data Scientist / Engineer?")

    submitted = st.form_submit_button("Submit")

    if submitted:
        required_fields = {
            "First Name": first_name,
            "Last Name": last_name,
            "Programming": prog_text,
            "Data Analysis": data_text,
            "ML Projects": ml_text,
            "ML Problem": ml_problem_text,
            "NLP": nlp_text,
            "Data Pipeline": pipeline_text,
            "Sharing Results": sharing_text,
            "Reflection": reflection_text
        }

        empty_fields = [name for name, value in required_fields.items() if not value.strip()]
        if empty_fields:
            st.warning(f"⚠️ Please fill in: {', '.join(empty_fields)}")
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

            # === Save locally first ===
            df = pd.DataFrame([responses])
            csv_path = "responses.csv"
            try:
                existing_df = pd.read_csv(csv_path)
                df = pd.concat([existing_df, df], ignore_index=True)
            except FileNotFoundError:
                pass
            df.to_csv(csv_path, index=False)

            # === Push to GitHub ===
            try:
                repo_path = os.getcwd()  # dossier local du repo
                repo = Repo(repo_path)
                repo.git.add(csv_path)
                repo.index.commit(f"Add response from {first_name} {last_name}")
                origin = repo.remote(name="origin")
                origin.push("main")
                st.success(f"✅ Responses saved and pushed to GitHub (main branch)!")
            except Exception as e:
                st.error(f"⚠️ Error pushing to GitHub: {e}")

            # === Confirmation display ===
            st.balloons()
            st.markdown("### 📋 Your Submitted Responses:")
            st.dataframe(df.tail(1).reset_index(drop=True))
