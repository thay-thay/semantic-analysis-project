"""
Visualization Page for Semantic Analysis Project

This module provides comprehensive visualizations and analytics for all collected user responses.
It fetches data from GitHub, processes it, and displays interactive charts and insights.

Features:
- Global statistics on responses
- Job recommendation trends
- Competency distribution analysis
- User skill level comparisons
- Interactive Plotly visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import base64
from io import StringIO
from datetime import datetime

# Import semantic engine for analysis
from semantic_engine import run_semantic_analysis, load_reference_data

# === Configuration ===
GITHUB_REPO = "thay-thay/semantic-analysis-project"
FILE_PATH = "data/user_responses.csv"
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")


def fetch_responses_from_github():
    """Fetch all user responses from GitHub CSV file.
    
    Returns:
        pd.DataFrame: DataFrame with all user responses, or None if fetch fails
    """
    if not GITHUB_TOKEN:
        st.error("❌ GitHub token not configured.")
        return None
    
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
            df = pd.read_csv(StringIO(content))
            return df
        elif response.status_code == 404:
            st.warning("⚠️ No responses found yet. The CSV file doesn't exist.")
            return None
        else:
            st.error(f"❌ Error fetching file: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"❌ Exception occurred: {str(e)}")
        return None


def analyze_all_users(df):
    """Run semantic analysis on all users in the dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame with user responses
        
    Returns:
        list: List of results dictionaries for each user
    """
    all_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in df.iterrows():
        status_text.text(f"Analyzing user {idx + 1}/{len(df)}: {row['First_Name']} {row['Last_Name']}")
        
        # Prepare responses dict
        responses = row.to_dict()
        
        # Run analysis
        results, error = run_semantic_analysis(responses)
        
        if results:
            results['user_name'] = f"{row['First_Name']} {row['Last_Name']}"
            results['timestamp'] = row['Timestamp']
            all_results.append(results)
        
        progress_bar.progress((idx + 1) / len(df))
    
    status_text.empty()
    progress_bar.empty()
    
    return all_results


def show_visualisations():
    """Main function to display all visualizations."""
    
    st.markdown("# 📊 Global Visualizations & Analytics")
    st.markdown("---")
    
    # Fetch data
    with st.spinner("Fetching data from GitHub..."):
        df = fetch_responses_from_github()
    
    if df is None or len(df) == 0:
        st.info("📭 No data available yet. Submit some responses first!")
        return
    
    # Display basic stats
    st.markdown("## 📈 Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Responses", len(df))
    with col2:
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            latest = df['Timestamp'].max().strftime("%Y-%m-%d %H:%M")
            st.metric("Latest Response", latest)
    with col3:
        unique_users = df[['First_Name', 'Last_Name']].drop_duplicates()
        st.metric("Unique Users", len(unique_users))
    
    st.markdown("---")
    
    # Analyze all users
    st.markdown("## 🧠 Semantic Analysis Results")
    
    with st.spinner("Running semantic analysis on all responses..."):
        all_results = analyze_all_users(df)
    
    if not all_results:
        st.warning("⚠️ No valid analysis results.")
        return
    
    # === 1. TOP JOBS DISTRIBUTION ===
    st.markdown("### 💼 Most Recommended Jobs")
    
    # Count top job recommendations
    top_jobs_counter = {}
    for result in all_results:
        if result['job_recommendations']:
            top_job = result['job_recommendations'][0]['job_title']
            top_jobs_counter[top_job] = top_jobs_counter.get(top_job, 0) + 1
    
    if top_jobs_counter:
        jobs_df = pd.DataFrame([
            {'Job Title': job, 'Count': count}
            for job, count in sorted(top_jobs_counter.items(), key=lambda x: x[1], reverse=True)
        ])
        
        fig_jobs = px.bar(
            jobs_df,
            x='Count',
            y='Job Title',
            orientation='h',
            title="Distribution of Top Job Recommendations",
            color='Count',
            color_continuous_scale='Viridis'
        )
        fig_jobs.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_jobs, use_container_width=True)
    
    # === 2. AVERAGE MATCH SCORES PER JOB ===
    st.markdown("### 🎯 Average Match Scores by Job")
    
    # Collect all job scores
    job_scores_data = []
    for result in all_results:
        for job in result['all_jobs']:
            job_scores_data.append({
                'Job Title': job['job_title'],
                'Match Score': job['match_score']
            })
    
    if job_scores_data:
        job_scores_df = pd.DataFrame(job_scores_data)
        avg_scores = job_scores_df.groupby('Job Title')['Match Score'].agg(['mean', 'std', 'count']).reset_index()
        avg_scores = avg_scores.sort_values('mean', ascending=False)
        
        fig_avg = px.bar(
            avg_scores.head(10),
            x='mean',
            y='Job Title',
            orientation='h',
            title="Top 10 Jobs by Average Match Score",
            labels={'mean': 'Average Match Score'},
            color='mean',
            color_continuous_scale='Blues'
        )
        fig_avg.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig_avg, use_container_width=True)
    
    # === 3. COMPETENCY BLOCK SCORES ===
    st.markdown("### 📊 Competency Block Analysis")
    
    # Aggregate block scores
    block_scores_data = []
    for result in all_results:
        for block, score in result['block_scores'].items():
            block_scores_data.append({
                'User': result['user_name'],
                'Block': block,
                'Score': score
            })
    
    if block_scores_data:
        blocks_df = pd.DataFrame(block_scores_data)
        
        # Average block scores across all users
        avg_blocks = blocks_df.groupby('Block')['Score'].mean().reset_index()
        avg_blocks = avg_blocks.sort_values('Score', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig_blocks_bar = px.bar(
                avg_blocks,
                x='Score',
                y='Block',
                orientation='h',
                title="Average Competency Block Scores",
                color='Score',
                color_continuous_scale='Greens',
                range_x=[0, 1]
            )
            fig_blocks_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_blocks_bar, use_container_width=True)
        
        with col2:
            # Radar chart
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=avg_blocks['Score'].tolist(),
                theta=avg_blocks['Block'].tolist(),
                fill='toself',
                name='Average Score',
                line_color='#2E86AB'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                showlegend=False,
                title="Competency Block Radar"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Box plot for distribution
        fig_box = px.box(
            blocks_df,
            x='Block',
            y='Score',
            title="Distribution of Block Scores Across All Users",
            color='Block'
        )
        fig_box.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_box, use_container_width=True)
    
    # === 4. OVERALL COVERAGE DISTRIBUTION ===
    st.markdown("### 🎯 Overall Coverage Distribution")
    
    coverage_data = [result['final_coverage'] for result in all_results]
    
    if coverage_data:
        fig_hist = px.histogram(
            x=coverage_data,
            nbins=20,
            title="Distribution of Overall Coverage Scores",
            labels={'x': 'Coverage Score', 'y': 'Number of Users'},
            color_discrete_sequence=['#A23B72']
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Coverage", f"{np.mean(coverage_data):.1%}")
        with col2:
            st.metric("Median Coverage", f"{np.median(coverage_data):.1%}")
        with col3:
            st.metric("Min Coverage", f"{np.min(coverage_data):.1%}")
        with col4:
            st.metric("Max Coverage", f"{np.max(coverage_data):.1%}")
    
    # === 5. TOP COMPETENCIES ===
    st.markdown("### 🏆 Most Relevant Competencies")
    
    # Load reference data to get competency names
    competencies_df, _, _, _ = load_reference_data()
    
    # Aggregate competency scores
    comp_scores_data = []
    for result in all_results:
        for comp_id, score in result['competency_scores'].items():
            comp_text = competencies_df[competencies_df['CompetencyID'] == comp_id]['CompetencyText'].values
            if len(comp_text) > 0:
                comp_scores_data.append({
                    'Competency': comp_text[0],
                    'Score': score,
                    'User': result['user_name']
                })
    
    if comp_scores_data:
        comps_df = pd.DataFrame(comp_scores_data)
        avg_comps = comps_df.groupby('Competency')['Score'].mean().reset_index()
        avg_comps = avg_comps.sort_values('Score', ascending=False).head(15)
        
        fig_comps = px.bar(
            avg_comps,
            x='Score',
            y='Competency',
            orientation='h',
            title="Top 15 Competencies by Average Score",
            color='Score',
            color_continuous_scale='Oranges',
            range_x=[0, max(avg_comps['Score']) * 1.1]
        )
        fig_comps.update_layout(showlegend=False, height=600)
        st.plotly_chart(fig_comps, use_container_width=True)
    
    # === 6. SKILL LEVELS (LIKERT) ===
    st.markdown("### 📊 Self-Reported Skill Levels")
    
    if 'Git_Level' in df.columns and 'Presentation_Level' in df.columns:
        skill_data = pd.DataFrame({
            'Git & Collaboration': df['Git_Level'],
            'Presentation Skills': df['Presentation_Level']
        })
        
        # Melt for easier plotting
        skill_melted = skill_data.melt(var_name='Skill', value_name='Level')
        
        fig_skills = px.violin(
            skill_melted,
            x='Skill',
            y='Level',
            box=True,
            points='all',
            title="Distribution of Self-Reported Skills",
            color='Skill'
        )
        fig_skills.update_layout(showlegend=False)
        st.plotly_chart(fig_skills, use_container_width=True)
        
        # Average levels
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Git Level", f"{df['Git_Level'].mean():.2f} / 5")
        with col2:
            st.metric("Average Presentation Level", f"{df['Presentation_Level'].mean():.2f} / 5")
    
    # === 7. USER COMPARISON TABLE ===
    st.markdown("### 👥 User Comparison")
    
    comparison_data = []
    for result in all_results:
        top_job = result['job_recommendations'][0] if result['job_recommendations'] else None
        comparison_data.append({
            'User': result['user_name'],
            'Coverage': f"{result['final_coverage']:.1%}",
            'Top Job': top_job['job_title'] if top_job else 'N/A',
            'Match Score': f"{top_job['match_score']:.1%}" if top_job else 'N/A',
            'Timestamp': result['timestamp']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # === 8. EXPORT OPTIONS ===
    st.markdown("---")
    st.markdown("### 💾 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Full Analysis (CSV)"):
            # Create detailed export
            export_data = []
            for result in all_results:
                top_job = result['job_recommendations'][0] if result['job_recommendations'] else {}
                export_data.append({
                    'User': result['user_name'],
                    'Timestamp': result['timestamp'],
                    'Overall_Coverage': result['final_coverage'],
                    'Top_Job': top_job.get('job_title', 'N/A'),
                    'Top_Job_Score': top_job.get('match_score', 0),
                    **{f"Block_{k}": v for k, v in result['block_scores'].items()}
                })
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        st.info("💡 Use the download button to export all analysis results for further processing.")
    
    st.markdown("---")
    st.success("✅ All visualizations loaded successfully!")
