# ============================================================
# 🚀 run_pipeline.py — The Main Script (Runs Everything!)
# ============================================================
# This is the "master" script that runs the ENTIRE ML pipeline
# from start to finish. Just run this one file and it will:
#
#   1. Load your data
#   2. Clean & prepare it
#   3. Train 3 ML models
#   4. Evaluate & compare them
#   5. Run a fairness/bias audit
#   6. Create all visualizations
#   7. Save everything
#
# HOW TO RUN:
#   python run_pipeline.py
# ============================================================

import sys
import os

# Add the project root to Python's path so it can find our src/ modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_collection import load_data, validate_data, get_data_summary
from src.preprocessing import infer_gender, engineer_features, prepare_for_modeling
from src.model import train_models, evaluate_models, save_models, get_comparison_table
from src.bias_audit import full_bias_report
from src.visualize import create_all_visualizations

import joblib


def run_full_pipeline():
    """
    Run the complete ML pipeline end-to-end.
    
    This function is the "main highway" of our project.
    It calls each step in order, passing data from one to the next,
    like an assembly line in a factory.
    """
    
    print("=" * 60)
    print("  🏦 LOAN APPROVAL BIAS AUDITOR — Full Pipeline")
    print("  📋 Predicting approvals & checking for fairness")
    print("=" * 60)
    
    # ==========================================
    # STEP 1: LOAD THE DATA
    # ==========================================
    print("\n" + "━" * 50)
    print("📂 STEP 1: Loading Data")
    print("━" * 50)
    
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loan_approval.csv")
    df = load_data(csv_path)
    validation_report = validate_data(df)
    summary = get_data_summary(df)
    
    # ==========================================
    # STEP 2: PREPROCESS THE DATA
    # ==========================================
    print("\n" + "━" * 50)
    print("🔧 STEP 2: Preprocessing & Feature Engineering")
    print("━" * 50)
    
    df = infer_gender(df)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_for_modeling(df)
    
    # ==========================================
    # STEP 3: TRAIN ML MODELS
    # ==========================================
    print("\n" + "━" * 50)
    print("🤖 STEP 3: Training ML Models")
    print("━" * 50)
    
    models = train_models(X_train, y_train)
    
    # ==========================================
    # STEP 4: EVALUATE & COMPARE MODELS
    # ==========================================
    print("\n" + "━" * 50)
    print("📊 STEP 4: Evaluating Models")
    print("━" * 50)
    
    results, best_model_name = evaluate_models(models, X_test, y_test)
    
    # Print comparison table
    comparison = get_comparison_table(results)
    print("\n📋 Model Comparison Table:")
    print(comparison.to_string(index=False))
    
    # ==========================================
    # STEP 5: BIAS & FAIRNESS AUDIT
    # ==========================================
    print("\n" + "━" * 50)
    print("⚖️  STEP 5: Fairness & Bias Audit")
    print("━" * 50)
    
    best_model = models[best_model_name]
    bias_report = full_bias_report(df, best_model, X_test, y_test, feature_names)
    
    # ==========================================
    # STEP 6: CREATE VISUALIZATIONS
    # ==========================================
    print("\n" + "━" * 50)
    print("🎨 STEP 6: Creating Visualizations")
    print("━" * 50)
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    viz_paths = create_all_visualizations(
        df, results, best_model, X_test, feature_names, output_dir
    )
    
    # ==========================================
    # STEP 7: SAVE EVERYTHING
    # ==========================================
    print("\n" + "━" * 50)
    print("💾 STEP 7: Saving Models & Data")
    print("━" * 50)
    
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    save_models(models, models_dir)
    
    # Save the scaler and feature names (needed for the dashboard)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    joblib.dump(feature_names, os.path.join(models_dir, 'feature_names.joblib'))
    print(f"💾 Saved: scaler and feature names")
    
    # Save the processed dataframe
    processed_path = os.path.join(output_dir, 'processed_data.csv')
    df.to_csv(processed_path, index=False)
    print(f"💾 Saved: {processed_path}")
    
    # ==========================================
    # DONE!
    # ==========================================
    print("\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLETE!")
    print(f"  🏆 Best Model: {best_model_name}")
    print(f"  📊 Charts saved in: {output_dir}/")
    print(f"  🤖 Models saved in: {models_dir}/")
    print(f"\n  🌐 To launch the dashboard, run:")
    print(f"     streamlit run app.py")
    print("=" * 60)
    
    return {
        'df': df,
        'models': models,
        'results': results,
        'best_model_name': best_model_name,
        'bias_report': bias_report,
        'feature_names': feature_names,
    }


# ============================================================
# This runs the pipeline when you execute:  python run_pipeline.py
# ============================================================
if __name__ == "__main__":
    run_full_pipeline()
