# ============================================================
# ⚖️ bias_audit.py — Step 4: Checking for Fairness & Bias
# ============================================================
# THIS IS WHAT MAKES THIS PROJECT UNIQUE!
# 
# We don't just predict loan approvals — we check if the 
# predictions (and the original data) are FAIR across groups.
#
# Questions we answer:
# - Are men and women approved at the same rate?
# - Does the model treat both genders equally?
# - What would happen if we flipped someone's gender?
# ============================================================

import pandas as pd
import numpy as np


def compute_approval_rates(df):
    """
    Calculate approval rates broken down by different groups.
    
    This is the simplest fairness check: just count what % of 
    each group got approved. If one group is 90% and another 
    is 40%, something might be unfair.
    """
    print("\n⚖️  Analyzing approval rates by group...")
    print("=" * 50)
    
    results = {}
    
    # --- By Gender ---
    if 'gender' in df.columns:
        print("\n👤 Approval rates by GENDER:")
        gender_stats = df.groupby('gender')['loan_approved'].agg(['mean', 'count'])
        gender_stats.columns = ['approval_rate', 'count']
        gender_stats['approval_rate'] = gender_stats['approval_rate'] * 100
        for g, row in gender_stats.iterrows():
            print(f"   {g}: {row['approval_rate']:.1f}% approved (n={int(row['count'])})")
        results['gender'] = gender_stats
    
    # --- By Credit Category ---
    if 'credit_category' in df.columns:
        print("\n💳 Approval rates by CREDIT CATEGORY:")
        credit_stats = df.groupby('credit_category')['loan_approved'].agg(['mean', 'count'])
        credit_stats.columns = ['approval_rate', 'count']
        credit_stats['approval_rate'] = credit_stats['approval_rate'] * 100
        for cat in ['Poor', 'Fair', 'Good', 'Excellent']:
            if cat in credit_stats.index:
                row = credit_stats.loc[cat]
                print(f"   {cat}: {row['approval_rate']:.1f}% (n={int(row['count'])})")
        results['credit_category'] = credit_stats
    
    # --- By Income Bracket ---
    if 'income_bracket' in df.columns:
        print("\n💰 Approval rates by INCOME BRACKET:")
        income_stats = df.groupby('income_bracket', observed=True)['loan_approved'].agg(['mean', 'count'])
        income_stats.columns = ['approval_rate', 'count']
        income_stats['approval_rate'] = income_stats['approval_rate'] * 100
        for bracket, row in income_stats.iterrows():
            print(f"   {bracket}: {row['approval_rate']:.1f}% (n={int(row['count'])})")
        results['income_bracket'] = income_stats
    
    return results


def disparate_impact_analysis(df):
    """
    Calculate the Disparate Impact Ratio (the "80% rule").
    
    WHAT IS DISPARATE IMPACT?
    The U.S. EEOC uses the "80% rule": if a protected group's 
    approval rate is less than 80% of the most-approved group's 
    rate, there may be illegal discrimination.
    
    Example:
      Male approval rate: 50%
      Female approval rate: 35%
      Ratio: 35/50 = 0.70 → Below 0.80 → ⚠️ Potential bias!
    """
    print("\n📏 Disparate Impact Analysis (80% Rule)...")
    print("=" * 50)
    
    results = {}
    
    if 'gender' in df.columns:
        gender_rates = df.groupby('gender')['loan_approved'].mean()
        
        # Only compare Male vs Female (skip Unknown)
        male_rate = gender_rates.get('Male', 0)
        female_rate = gender_rates.get('Female', 0)
        
        if male_rate > 0 and female_rate > 0:
            # Ratio = disadvantaged group / advantaged group
            if male_rate >= female_rate:
                ratio = female_rate / male_rate
                disadvantaged = 'Female'
                advantaged = 'Male'
            else:
                ratio = male_rate / female_rate
                disadvantaged = 'Male'
                advantaged = 'Female'
            
            print(f"   {advantaged} approval rate: {max(male_rate, female_rate)*100:.1f}%")
            print(f"   {disadvantaged} approval rate: {min(male_rate, female_rate)*100:.1f}%")
            print(f"   Disparate Impact Ratio: {ratio:.3f}")
            
            if ratio >= 0.8:
                print("   ✅ PASSES the 80% rule — no significant gender bias detected")
            else:
                print("   ⚠️  FAILS the 80% rule — potential gender bias detected!")
            
            results['gender_di_ratio'] = ratio
            results['advantaged_group'] = advantaged
            results['disadvantaged_group'] = disadvantaged
    
    return results


def what_if_analysis(model, X_test, feature_names):
    """
    "What-If" analysis: What happens if we flip someone's gender?
    
    This is a powerful fairness test. We take each person's data,
    change ONLY their gender (keep everything else the same), 
    and check if the model's prediction changes.
    
    If the prediction changes just because of gender, the model
    might be biased.
    """
    print("\n🔄 What-If Analysis: Flipping gender...")
    print("=" * 50)
    
    if 'gender_encoded' not in feature_names:
        print("   ⚠️ No gender feature found, skipping.")
        return {}
    
    gender_idx = feature_names.index('gender_encoded')
    
    # Get original predictions
    original_preds = model.predict(X_test)
    original_probs = model.predict_proba(X_test)[:, 1]
    
    # Create a copy and flip gender (Male↔Female)
    X_flipped = X_test.copy()
    # In our encoding: Male=1, Female=0
    # We'll swap: where it was Male, make it Female and vice versa
    col = 'gender_encoded'
    X_flipped[col] = X_flipped[col].apply(
        lambda x: 0 if x == 1 else (1 if x == 0 else x)
    )
    
    # Get new predictions with flipped gender
    flipped_preds = model.predict(X_flipped)
    flipped_probs = model.predict_proba(X_flipped)[:, 1]
    
    # How many predictions changed?
    changed = (original_preds != flipped_preds).sum()
    total = len(original_preds)
    avg_prob_change = np.mean(np.abs(original_probs - flipped_probs))
    
    print(f"   Total test samples: {total}")
    print(f"   Predictions that CHANGED when gender was flipped: {changed} ({changed/total*100:.1f}%)")
    print(f"   Average probability shift: {avg_prob_change:.4f}")
    
    if changed / total < 0.05:
        print("   ✅ Model is mostly gender-neutral (< 5% predictions changed)")
    elif changed / total < 0.15:
        print("   ⚠️  Moderate gender sensitivity (5-15% predictions changed)")
    else:
        print("   🚨 High gender sensitivity (> 15% predictions changed) — potential bias!")
    
    return {
        'predictions_changed': changed,
        'total_samples': total,
        'change_rate': changed / total,
        'avg_probability_shift': avg_prob_change,
    }


def full_bias_report(df, model, X_test, y_test, feature_names):
    """Run all bias checks and compile a complete fairness report."""
    print("\n" + "=" * 60)
    print("   ⚖️  COMPLETE FAIRNESS & BIAS AUDIT REPORT")
    print("=" * 60)
    
    report = {}
    report['approval_rates'] = compute_approval_rates(df)
    report['disparate_impact'] = disparate_impact_analysis(df)
    report['what_if'] = what_if_analysis(model, X_test, feature_names)
    
    print("\n" + "=" * 60)
    print("   ✅ Bias audit complete!")
    print("=" * 60 + "\n")
    
    return report
