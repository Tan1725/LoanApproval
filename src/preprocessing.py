# ============================================================
# 🔧 preprocessing.py — Step 2: Cleaning & Preparing the Data
# ============================================================
# Raw data is messy — this file cleans it, infers gender from 
# names, creates smart new features, and prepares everything 
# for our ML models.
# ============================================================

import pandas as pd
import numpy as np
import gender_guesser.detector as gender
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def infer_gender(df):
    """Guess gender from first names using the gender_guesser library."""
    print("🔍 Inferring gender from applicant names...")
    detector = gender.Detector()
    
    def get_gender(full_name):
        parts = str(full_name).split()
        titles = ['mr.', 'mrs.', 'ms.', 'dr.', 'mr', 'mrs', 'ms', 'dr']
        first_name = parts[0]
        if first_name.lower() in titles and len(parts) > 1:
            first_name = parts[1]
        result = detector.get_gender(first_name)
        if result in ['male', 'mostly_male']:
            return 'Male'
        elif result in ['female', 'mostly_female']:
            return 'Female'
        else:
            return 'Unknown'
    
    df['gender'] = df['name'].apply(get_gender)
    gender_counts = df['gender'].value_counts()
    print("   Gender distribution:")
    for g, count in gender_counts.items():
        print(f"   - {g}: {count} ({count/len(df)*100:.1f}%)")
    return df


def engineer_features(df):
    """
    Create new features from existing columns.
    
    Feature engineering = creating smarter columns that help the 
    model make better predictions. For example, debt_to_income 
    ratio tells us how much someone wants to borrow vs what they earn.
    """
    print("\n⚙️  Engineering new features...")
    
    # Debt-to-Income: What fraction of income is the loan?
    df['debt_to_income'] = df['loan_amount'] / df['income']
    print("   ✅ debt_to_income (loan_amount / income)")
    
    # Income per year of experience
    df['income_per_exp'] = df['income'] / (df['years_employed'] + 1)
    print("   ✅ income_per_exp (income / years_employed)")
    
    # Credit score category: Poor / Fair / Good / Excellent
    def categorize_credit(score):
        if score < 580: return 'Poor'
        elif score < 670: return 'Fair'
        elif score < 740: return 'Good'
        else: return 'Excellent'
    
    df['credit_category'] = df['credit_score'].apply(categorize_credit)
    print("   ✅ credit_category (Poor/Fair/Good/Excellent)")
    
    # Loan relative to points
    df['loan_to_points'] = df['loan_amount'] / (df['points'] + 1)
    print("   ✅ loan_to_points (loan_amount / points)")
    
    # Income bracket for analysis
    df['income_bracket'] = pd.cut(
        df['income'],
        bins=[0, 40000, 70000, 100000, 130000, 200000],
        labels=['<40K', '40K-70K', '70K-100K', '100K-130K', '130K+']
    )
    print("   ✅ income_bracket (income ranges)")
    print(f"   📊 Total features now: {len(df.columns)}")
    return df


def prepare_for_modeling(df):
    """
    Final prep: select features, encode text→numbers, split train/test, scale.
    
    WHY SPLIT? We train on 80% and test on the unseen 20% to check 
    if the model truly learned patterns (not just memorized the data).
    
    WHY SCALE? Income (30K-150K) and credit score (300-850) are on 
    different scales. Scaling puts them on similar ranges so models 
    aren't confused.
    """
    print("\n📦 Preparing data for ML models...")
    
    numeric_features = [
        'income', 'credit_score', 'loan_amount', 'years_employed',
        'points', 'debt_to_income', 'income_per_exp', 'loan_to_points',
    ]
    
    # Convert gender text to numbers (Label Encoding)
    gender_map = {'Male': 1, 'Female': 0, 'Unknown': 2}
    df['gender_encoded'] = df['gender'].map(gender_map)
    numeric_features.append('gender_encoded')
    
    feature_names = numeric_features
    X = df[feature_names].copy()
    y = df['loan_approved'].astype(int)  # True/False → 1/0
    
    # Fill any missing values with column median
    X = X.fillna(X.median())
    
    # Split: 80% train, 20% test, stratify keeps approval ratio balanced
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training: {len(X_train)} | Testing: {len(X_test)}")
    
    # Scale features to mean=0, std=1
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )
    
    print("   ✅ Data ready for modeling!\n")
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler
