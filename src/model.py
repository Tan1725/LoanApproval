# ============================================================
# 🤖 model.py — Step 3: Training ML Models
# ============================================================
# This is the core ML step! We train 3 different models and 
# compare which one predicts loan approvals best.
#
# Models used:
#   1. Logistic Regression — Simple, interpretable, great baseline
#   2. Random Forest — Ensemble of decision trees, very robust
#   3. XGBoost — Most powerful, often wins competitions
# ============================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, 
    confusion_matrix
)
from xgboost import XGBClassifier
import joblib  # For saving trained models to disk
import os


def train_models(X_train, y_train):
    """
    Train 3 different ML models on the training data.
    
    WHY 3 MODELS?
    We try multiple approaches because no single model is always best.
    By comparing them, we can pick the winner for our specific data.
    
    Returns a dictionary: {'model_name': trained_model_object}
    """
    print("🤖 Training ML models...")
    print("=" * 50)
    
    models = {}
    
    # --- Model 1: Logistic Regression ---
    # Think of it as drawing a line between "approved" and "denied".
    # Simple but effective. Great for understanding which features matter.
    print("\n1️⃣  Logistic Regression (the simple, interpretable one)...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)  # .fit() = "learn from this data"
    models['Logistic Regression'] = lr
    print("   ✅ Trained!")
    
    # --- Model 2: Random Forest ---
    # Imagine asking 100 different "decision tree" experts to vote.
    # Each tree looks at a random subset of features.
    # The final answer = majority vote. Very robust!
    print("\n2️⃣  Random Forest (100 decision trees voting together)...")
    rf = RandomForestClassifier(
        n_estimators=100,   # Use 100 trees
        max_depth=10,       # Each tree can be max 10 levels deep
        random_state=42     # Reproducible results
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    print("   ✅ Trained!")
    
    # --- Model 3: XGBoost ---
    # The heavyweight champion. Builds trees one at a time,
    # where each new tree tries to fix the mistakes of all previous trees.
    print("\n3️⃣  XGBoost (the competition-winning powerhouse)...")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,   # How quickly it learns (slower = more careful)
        random_state=42,
        eval_metric='logloss',
        verbosity=0          # Don't print training logs
    )
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb
    print("   ✅ Trained!")
    
    print(f"\n✅ All {len(models)} models trained successfully!")
    return models


def evaluate_models(models, X_test, y_test):
    """
    Test each model on data it has NEVER seen before.
    
    METRICS EXPLAINED:
    - Accuracy: What % of predictions were correct overall?
    - Precision: When the model says "approved", how often is it right?
    - Recall: Of all actual approvals, how many did the model catch?
    - F1-Score: The balance between precision and recall (higher = better)
    - AUC-ROC: Overall model quality on a 0-1 scale (1 = perfect)
    """
    print("\n📊 Evaluating models on test data...")
    print("=" * 50)
    
    results = {}
    
    for name, model in models.items():
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability of approval
        
        # Calculate all metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred,
            'probabilities': y_prob,
        }
        results[name] = metrics
        
        # Print results in a readable format
        print(f"\n📋 {name}:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score']:.4f}")
        print(f"   AUC-ROC:   {metrics['auc_roc']:.4f}")
        
        # Confusion matrix explained
        cm = metrics['confusion_matrix']
        print(f"   Confusion Matrix:")
        print(f"     True Negatives (correctly denied):  {cm[0][0]}")
        print(f"     False Positives (wrongly approved):  {cm[0][1]}")
        print(f"     False Negatives (wrongly denied):    {cm[1][0]}")
        print(f"     True Positives (correctly approved): {cm[1][1]}")
    
    # Find the best model
    best_name = max(results, key=lambda x: results[x]['f1_score'])
    print(f"\n🏆 Best model: {best_name} (F1-Score: {results[best_name]['f1_score']:.4f})")
    
    return results, best_name


def save_models(models, output_dir="models"):
    """Save trained models to disk so we can reuse them later."""
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        filename = os.path.join(output_dir, f"{name.lower().replace(' ', '_')}.joblib")
        joblib.dump(model, filename)
        print(f"💾 Saved: {filename}")


def get_comparison_table(results):
    """Create a nice comparison table of all models."""
    rows = []
    for name, metrics in results.items():
        rows.append({
            'Model': name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'AUC-ROC': f"{metrics['auc_roc']:.4f}",
        })
    return pd.DataFrame(rows)
