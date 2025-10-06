import streamlit as st
import pandas as pd
from datetime import datetime

# === Page configuration ===
st.set_page_config(
    page_title="Semantic Analysis Project",
    page_icon="🧠",
    layout="wide"
)

# === Custom CSS for modern font, colors, and sliders ===
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

        /* Remove slider label above (tooltip only remains) */
        div[data-baseweb="slider"] > div > div > div > div > div[role="slider"] + div {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# === Header (without logo) ===
st.markdown("""
<div>
    <div class="header-title">Project – Semantic Analysis</div>
    <div class="header-subtitle">Semantic Analysis for Competency Mapping and Job Profile Recommendation</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# === Form Section ===
with st.form("skills_form"):
    # === Basic Info ===
    first_name = st.text_input("First Name", placeholder="Enter your first name")
    last_name = st.text_input("Last Name", placeholder="Enter your last name")

    # === Experience Questions ===
    prog_text = st.text_area(
        "Describe your experience with programming. What languages or tools have you used most?",
        placeholder="Ex: I mostly use Python and SQL, and I work with Git and OOP concepts."
    )

    data_text = st.text_area(
        "Explain how you typically analyze a dataset before building a model.",
        placeholder="Ex: I clean the data, perform EDA, visualize distributions, and calculate statistics."
    )

    ml_text = st.text_area(
        "Tell us about a project where you applied machine learning techniques. What did you do and what tools did you use?",
        placeholder="Ex: I built a regression model using scikit-learn and evaluated it with cross-validation."
    )

    ml_problem_text = st.text_area(
        "How would you approach designing a machine learning model for predicting customer churn?",
        placeholder="Ex: I would perform feature engineering, select a model, train, and evaluate it."
    )

    nlp_text = st.text_area(
        "Have you ever worked with text data (NLP)? What techniques or libraries did you use?",
        placeholder="Ex: I tokenized text, used embeddings, transformers, sentiment analysis, and NER."
    )

    pipeline_text = st.text_area(
        "Explain a time when you built or maintained a data pipeline. What tools or frameworks were involved?",
        placeholder="Ex: I implemented an ETL pipeline using Airflow for batch processing."
    )

    sharing_text = st.text_area(
        "How do you usually share the results of your data analysis with others?",
        placeholder="Ex: I create dashboards, visualizations, and prepare presentations to explain insights."
    )

    # === Sliders ===
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

    # === Reflection ===
    reflection_text = st.text_area(
        "In your opinion, what makes someone a strong Data Scientist / Engineer?",
        placeholder="Ex: Strong problem-solving, communication skills, and mastery of tools."
    )

    # === Submit Button ===
    submitted = st.form_submit_button("Submit")

    if submitted:
        # Vérifier les champs obligatoires
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

        empty_fields = [name for name, value in required_fields.items() if not value or value.strip() == ""]

        if empty_fields:
            st.warning(f"⚠️ Please fill in all the required fields before submitting: {', '.join(empty_fields)}")
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

            # === Save to CSV ===
            df = pd.DataFrame([responses])
            try:
                existing_df = pd.read_csv("responses.csv")
                df = pd.concat([existing_df, df], ignore_index=True)
            except FileNotFoundError:
                pass

            df.to_csv("responses.csv", index=False)
            st.success("✅ Your responses have been successfully saved!")
            st.balloons()
