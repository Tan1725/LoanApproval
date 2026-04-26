# ============================================================
# 📊 visualize.py — Step 5: Charts, Graphs & SHAP Explanations
# ============================================================
# This file creates all the visual outputs for our project.
# Good visualizations tell the story of your data and make 
# your project impressive in presentations.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (works without display)
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os


# Set a clean, modern style for all charts
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_approval_by_gender(df, output_dir="outputs"):
    """Bar chart: Approval rates by gender."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Chart 1: Counts
    gender_counts = df.groupby(['gender', 'loan_approved']).size().unstack(fill_value=0)
    gender_counts.plot(kind='bar', ax=axes[0], color=['#e74c3c', '#2ecc71'])
    axes[0].set_title('Loan Decisions by Gender (Counts)', fontweight='bold')
    axes[0].set_xlabel('Gender')
    axes[0].set_ylabel('Number of Applications')
    axes[0].legend(['Denied', 'Approved'])
    axes[0].tick_params(axis='x', rotation=0)
    
    # Chart 2: Rates
    gender_rates = df.groupby('gender')['loan_approved'].mean() * 100
    bars = gender_rates.plot(kind='bar', ax=axes[1], color=['#3498db', '#2ecc71', '#95a5a6'])
    axes[1].set_title('Approval Rate by Gender (%)', fontweight='bold')
    axes[1].set_xlabel('Gender')
    axes[1].set_ylabel('Approval Rate (%)')
    axes[1].tick_params(axis='x', rotation=0)
    # Add percentage labels on bars
    for i, (idx, val) in enumerate(gender_rates.items()):
        axes[1].text(i, val + 1, f'{val:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'approval_by_gender.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: {path}")
    return path


def plot_approval_by_credit(df, output_dir="outputs"):
    """Bar chart: Approval rates by credit score category."""
    os.makedirs(output_dir, exist_ok=True)
    
    order = ['Poor', 'Fair', 'Good', 'Excellent']
    existing = [c for c in order if c in df['credit_category'].values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rates = df.groupby('credit_category')['loan_approved'].mean().reindex(existing) * 100
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71'][:len(existing)]
    rates.plot(kind='bar', ax=ax, color=colors)
    ax.set_title('Approval Rate by Credit Score Category', fontweight='bold', fontsize=14)
    ax.set_xlabel('Credit Category')
    ax.set_ylabel('Approval Rate (%)')
    ax.tick_params(axis='x', rotation=0)
    for i, val in enumerate(rates):
        ax.text(i, val + 1, f'{val:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'approval_by_credit.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: {path}")
    return path


def plot_feature_distributions(df, output_dir="outputs"):
    """Histograms of key features, colored by approval status."""
    os.makedirs(output_dir, exist_ok=True)
    
    features = ['income', 'credit_score', 'loan_amount', 'years_employed', 'points']
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, feat in enumerate(features):
        ax = axes[i]
        for approved, color, label in [(False, '#e74c3c', 'Denied'), (True, '#2ecc71', 'Approved')]:
            subset = df[df['loan_approved'] == approved][feat]
            ax.hist(subset, bins=25, alpha=0.6, color=color, label=label)
        ax.set_title(f'{feat}', fontweight='bold')
        ax.legend(fontsize=9)
    
    # Hide the extra subplot
    axes[5].set_visible(False)
    
    fig.suptitle('Feature Distributions by Approval Status', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'feature_distributions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: {path}")
    return path


def plot_correlation_heatmap(df, output_dir="outputs"):
    """Heatmap showing how features correlate with each other."""
    os.makedirs(output_dir, exist_ok=True)
    
    numeric_cols = ['income', 'credit_score', 'loan_amount', 'years_employed', 
                    'points', 'debt_to_income', 'loan_approved']
    # Convert loan_approved to int for correlation
    plot_df = df[numeric_cols].copy()
    plot_df['loan_approved'] = plot_df['loan_approved'].astype(int)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = plot_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Show only bottom triangle
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, ax=ax, square=True, linewidths=1)
    ax.set_title('Feature Correlation Heatmap', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: {path}")
    return path


def plot_model_comparison(results, output_dir="outputs"):
    """Bar chart comparing all 3 models across metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    model_names = list(results.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for i, name in enumerate(model_names):
        values = [results[name][m] for m in metrics_to_plot]
        bars = ax.bar(x + i * width, values, width, label=name, color=colors[i], alpha=0.85)
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'])
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: {path}")
    return path


def plot_confusion_matrices(results, output_dir="outputs"):
    """Confusion matrix heatmaps for each model."""
    os.makedirs(output_dir, exist_ok=True)
    
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, metrics) in zip(axes, results.items()):
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Denied', 'Approved'],
                    yticklabels=['Denied', 'Approved'])
        ax.set_title(f'{name}', fontweight='bold')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
    
    fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: {path}")
    return path


def plot_shap_analysis(model, X_test, feature_names, output_dir="outputs"):
    """
    SHAP analysis — explains WHY the model made each prediction.
    
    SHAP (SHapley Additive exPlanations) comes from game theory.
    It tells you how much each feature pushed the prediction 
    toward "approved" or "denied" for EACH individual person.
    
    Red = pushed toward approval, Blue = pushed toward denial.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("   🔬 Running SHAP analysis (this may take a moment)...")
    
    # Create SHAP explainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except Exception:
        # Fallback for non-tree models
        explainer = shap.LinearExplainer(model, X_test)
        shap_values = explainer.shap_values(X_test)
    
    # Plot 1: Summary plot (most important features)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                      show=False, plot_size=(10, 6))
    plt.title('SHAP Feature Importance\n(How each feature affects predictions)', 
              fontweight='bold', fontsize=13)
    plt.tight_layout()
    path1 = os.path.join(output_dir, 'shap_summary.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: {path1}")
    
    # Plot 2: Bar plot (average feature importance)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      plot_type='bar', show=False, plot_size=(10, 6))
    plt.title('Average Feature Importance (SHAP)', fontweight='bold', fontsize=13)
    plt.tight_layout()
    path2 = os.path.join(output_dir, 'shap_importance.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: {path2}")
    
    return path1, path2, shap_values


def create_all_visualizations(df, results, best_model, X_test, feature_names, output_dir="outputs"):
    """Generate all charts at once."""
    print("\n🎨 Creating visualizations...")
    print("=" * 50)
    
    paths = {}
    paths['gender'] = plot_approval_by_gender(df, output_dir)
    paths['credit'] = plot_approval_by_credit(df, output_dir)
    paths['distributions'] = plot_feature_distributions(df, output_dir)
    paths['correlation'] = plot_correlation_heatmap(df, output_dir)
    paths['comparison'] = plot_model_comparison(results, output_dir)
    paths['confusion'] = plot_confusion_matrices(results, output_dir)
    
    try:
        shap1, shap2, shap_vals = plot_shap_analysis(best_model, X_test, feature_names, output_dir)
        paths['shap_summary'] = shap1
        paths['shap_importance'] = shap2
    except Exception as e:
        print(f"   ⚠️ SHAP analysis skipped: {e}")
    
    print(f"\n✅ All visualizations saved to '{output_dir}/' folder!")
    return paths
