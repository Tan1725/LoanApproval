# ============================================================
# 🌐 app.py — Streamlit Dashboard
# ============================================================
# This creates a beautiful, interactive web app for your project.
# Run it with:  streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Page Configuration ----
st.set_page_config(
    page_title="Loan Approval Bias Auditor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS for a premium look ----
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border-radius: 12px; padding: 1.2rem; border: 1px solid #667eea44;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #667eea; }
    .metric-label { font-size: 0.9rem; color: #888; margin-top: 0.3rem; }
    .section-header {
        font-size: 1.5rem; font-weight: 700; color: #333;
        border-bottom: 3px solid #667eea; padding-bottom: 0.5rem;
        margin: 2rem 0 1rem;
    }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e, #16213e); }
    div[data-testid="stSidebar"] * { color: #eee !important; }
</style>
""", unsafe_allow_html=True)

# ---- Helper: load resources ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_processed_data():
    path = os.path.join(BASE_DIR, "outputs", "processed_data.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_resource
def load_models():
    models = {}
    models_dir = os.path.join(BASE_DIR, "models")
    for fname in ['logistic_regression.joblib', 'random_forest.joblib', 'xgboost.joblib']:
        path = os.path.join(models_dir, fname)
        if os.path.exists(path):
            name = fname.replace('.joblib', '').replace('_', ' ').title()
            models[name] = joblib.load(path)
    return models

# ---- Load data & models ----
df = load_processed_data()
models = load_models()

# ---- Sidebar ----
with st.sidebar:
    st.markdown("## 🏦 Navigation")
    page = st.radio("Go to:", [
        "📊 Overview",
        "🤖 Model Results",
        "⚖️ Bias Audit",
        "🔮 Predict New Loan",
        "📸 All Charts"
    ])
    st.markdown("---")
    st.markdown("### About This Project")
    st.markdown(
        "This app predicts loan approvals and "
        "audits the model for **gender bias** using "
        "fairness metrics and SHAP explainability."
    )

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "📊 Overview":
    st.markdown('<div class="main-header">🏦 Loan Approval Bias Auditor</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#888;'>Predicting loan approvals & checking for fairness with ML</p>", unsafe_allow_html=True)
    
    if df is not None:
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Total Applications</div></div>', unsafe_allow_html=True)
        with col2:
            rate = df['loan_approved'].mean() * 100
            st.markdown(f'<div class="metric-card"><div class="metric-value">{rate:.1f}%</div><div class="metric-label">Approval Rate</div></div>', unsafe_allow_html=True)
        with col3:
            avg_income = df['income'].mean()
            st.markdown(f'<div class="metric-card"><div class="metric-value">${avg_income:,.0f}</div><div class="metric-label">Avg Income</div></div>', unsafe_allow_html=True)
        with col4:
            avg_credit = df['credit_score'].mean()
            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_credit:.0f}</div><div class="metric-label">Avg Credit Score</div></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">📋 Sample Data</div>', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True, height=400)
        
        # Gender distribution
        st.markdown('<div class="section-header">👤 Gender Distribution</div>', unsafe_allow_html=True)
        if 'gender' in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                gender_counts = df['gender'].value_counts()
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ['#667eea', '#764ba2', '#aaa'][:len(gender_counts)]
                gender_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=colors, startangle=90)
                ax.set_ylabel('')
                ax.set_title('Gender Split', fontweight='bold')
                st.pyplot(fig)
                plt.close()
            with col2:
                gender_approval = df.groupby('gender')['loan_approved'].mean() * 100
                fig, ax = plt.subplots(figsize=(6, 4))
                gender_approval.plot(kind='bar', ax=ax, color=['#667eea', '#764ba2', '#aaa'][:len(gender_approval)])
                ax.set_title('Approval Rate by Gender', fontweight='bold')
                ax.set_ylabel('Approval Rate (%)')
                ax.tick_params(axis='x', rotation=0)
                for i, v in enumerate(gender_approval):
                    ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
                st.pyplot(fig)
                plt.close()
    else:
        st.warning("⚠️ Run `python run_pipeline.py` first to generate the data!")

# ============================================================
# PAGE 2: MODEL RESULTS
# ============================================================
elif page == "🤖 Model Results":
    st.markdown('<div class="main-header">🤖 Model Performance</div>', unsafe_allow_html=True)
    
    # Show saved charts
    output_dir = os.path.join(BASE_DIR, "outputs")
    
    charts = {
        'Model Comparison': 'model_comparison.png',
        'Confusion Matrices': 'confusion_matrices.png',
        'SHAP Feature Importance': 'shap_importance.png',
        'SHAP Summary': 'shap_summary.png',
    }
    
    for title, filename in charts.items():
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
            st.image(path, use_container_width=True)

# ============================================================
# PAGE 3: BIAS AUDIT
# ============================================================
elif page == "⚖️ Bias Audit":
    st.markdown('<div class="main-header">⚖️ Fairness & Bias Audit</div>', unsafe_allow_html=True)
    
    if df is not None and 'gender' in df.columns:
        # Disparate Impact
        st.markdown('<div class="section-header">📏 Disparate Impact (80% Rule)</div>', unsafe_allow_html=True)
        st.markdown("""
        > **The 80% Rule**: If a group's approval rate is less than 80% of the 
        > most-approved group's rate, there may be discrimination.
        """)
        
        gender_rates = df.groupby('gender')['loan_approved'].mean()
        male_r = gender_rates.get('Male', 0)
        female_r = gender_rates.get('Female', 0)
        
        if male_r > 0 and female_r > 0:
            ratio = min(male_r, female_r) / max(male_r, female_r)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Male Approval Rate", f"{male_r*100:.1f}%")
            with col2:
                st.metric("Female Approval Rate", f"{female_r*100:.1f}%")
            with col3:
                st.metric("Disparate Impact Ratio", f"{ratio:.3f}", 
                          delta="PASS ✅" if ratio >= 0.8 else "FAIL ⚠️")
        
        # Charts
        output_dir = os.path.join(BASE_DIR, "outputs")
        for filename in ['approval_by_gender.png', 'approval_by_credit.png']:
            path = os.path.join(output_dir, filename)
            if os.path.exists(path):
                st.image(path, use_container_width=True)
        
        # Approval by credit + gender cross
        st.markdown('<div class="section-header">💳 Credit Category × Gender</div>', unsafe_allow_html=True)
        if 'credit_category' in df.columns:
            cross = df.groupby(['credit_category', 'gender'])['loan_approved'].mean().unstack() * 100
            st.dataframe(cross.style.format("{:.1f}%").background_gradient(cmap='RdYlGn'), use_container_width=True)
    else:
        st.warning("⚠️ Run `python run_pipeline.py` first!")

# ============================================================
# PAGE 4: PREDICT NEW LOAN
# ============================================================
elif page == "🔮 Predict New Loan":
    st.markdown('<div class="main-header">🔮 Predict a New Loan Application</div>', unsafe_allow_html=True)
    st.markdown("Enter applicant details below to get an instant prediction.")
    
    if models:
        max_income = int(df['income'].max()) if df is not None else 1000000
        max_loan = int(df['loan_amount'].max()) if df is not None else 500000

        col1, col2 = st.columns(2)
        with col1:
            income = st.number_input("💰 Annual Income ($)", min_value=0, max_value=max_income, value=min(75000, max_income), step=5000)
            credit_score = st.slider("💳 Credit Score", 300, 850, 650)
            loan_amount = st.number_input("🏠 Loan Amount ($)", min_value=0, max_value=max_loan, value=min(20000, max_loan), step=1000)
        with col2:
            years_employed = st.slider("📅 Years Employed", 0, 40, 10)
            points = st.slider("⭐ Points Score", 0.0, 100.0, 50.0, step=5.0)
            gender_input = st.selectbox("👤 Gender", ["Male", "Female"])
        
        if st.button("🔮 Predict Loan Decision", use_container_width=True):
            # Calculate engineered features
            debt_to_income = loan_amount / income
            income_per_exp = income / (years_employed + 1)
            loan_to_points = loan_amount / (points + 1)
            gender_encoded = 1 if gender_input == "Male" else 0
            
            # Load scaler
            scaler_path = os.path.join(BASE_DIR, "models", "scaler.joblib")
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                features = pd.DataFrame([[
                    income, credit_score, loan_amount, years_employed,
                    points, debt_to_income, income_per_exp, loan_to_points,
                    gender_encoded
                ]], columns=[
                    'income', 'credit_score', 'loan_amount', 'years_employed',
                    'points', 'debt_to_income', 'income_per_exp', 'loan_to_points',
                    'gender_encoded'
                ])
                features_scaled = scaler.transform(features)
                
                st.markdown("---")
                cols = st.columns(len(models))
                for i, (name, model) in enumerate(models.items()):
                    pred = model.predict(features_scaled)[0]
                    prob = model.predict_proba(features_scaled)[0][1]
                    with cols[i]:
                        if pred == 1:
                            st.success(f"**{name}**\n\n✅ APPROVED\n\nConfidence: {prob*100:.1f}%")
                        else:
                            st.error(f"**{name}**\n\n❌ DENIED\n\nApproval chance: {prob*100:.1f}%")
                
                # What-if: flip gender
                st.markdown('<div class="section-header">🔄 What-If: Flip Gender</div>', unsafe_allow_html=True)
                flipped_gender = 0 if gender_encoded == 1 else 1
                features_flipped = features.copy()
                features_flipped['gender_encoded'] = flipped_gender
                features_flipped_scaled = scaler.transform(features_flipped)
                
                other_gender = "Female" if gender_input == "Male" else "Male"
                cols2 = st.columns(len(models))
                for i, (name, model) in enumerate(models.items()):
                    orig_prob = model.predict_proba(features_scaled)[0][1]
                    flip_prob = model.predict_proba(features_flipped_scaled)[0][1]
                    diff = flip_prob - orig_prob
                    with cols2[i]:
                        st.metric(
                            f"{name}",
                            f"{flip_prob*100:.1f}% as {other_gender}",
                            delta=f"{diff*100:+.2f}% change"
                        )
            else:
                st.warning("Scaler not found. Run the pipeline first.")
    else:
        st.warning("⚠️ No models found. Run `python run_pipeline.py` first!")

# ============================================================
# PAGE 5: ALL CHARTS
# ============================================================
elif page == "📸 All Charts":
    st.markdown('<div class="main-header">📸 All Visualizations</div>', unsafe_allow_html=True)
    
    output_dir = os.path.join(BASE_DIR, "outputs")
    if os.path.exists(output_dir):
        images = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
        if images:
            for img_name in images:
                title = img_name.replace('.png', '').replace('_', ' ').title()
                st.markdown(f"### {title}")
                st.image(os.path.join(output_dir, img_name), use_container_width=True)
                st.markdown("---")
        else:
            st.info("No charts found. Run the pipeline first.")
    else:
        st.warning("⚠️ Run `python run_pipeline.py` first!")
