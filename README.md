# 🏦 Loan Approval Bias Auditor

An AI/ML project that predicts loan approvals and **audits the model for gender bias** using fairness metrics, SHAP explainability, and what-if analysis.

## What This Project Does

1. **Loads** your loan application data (2,001 real applications)
2. **Infers gender** from applicant names (demographic enrichment)
3. **Engineers smart features** like debt-to-income ratio
4. **Trains 3 ML models** and compares their performance
5. **Audits for bias** — checks if the model treats men and women fairly
6. **Explains predictions** using SHAP (why was this loan denied?)
7. **Deploys a dashboard** where you can test new applications live

## Quick Start

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the full pipeline
python run_pipeline.py

# Step 3: Launch the dashboard
streamlit run app.py
```

## Project Structure

```
📁 loanApproval project/
├── 📄 loan_approval.csv         ← Your raw data (2,001 loan applications)
├── 📄 requirements.txt          ← Python libraries needed
├── 📄 run_pipeline.py           ← 🚀 Main script — runs everything!
├── 📄 app.py                    ← 🌐 Streamlit dashboard
├── 📁 src/                      ← Source code modules
│   ├── data_collection.py       ← Step 1: Load & validate data
│   ├── preprocessing.py         ← Step 2: Clean, enrich & prepare data
│   ├── model.py                 ← Step 3: Train & evaluate ML models
│   ├── bias_audit.py            ← Step 4: Fairness & bias analysis
│   └── visualize.py             ← Step 5: Charts & SHAP plots
├── 📁 models/                   ← Saved trained models (created after running)
├── 📁 outputs/                  ← Charts & processed data (created after running)
└── 📄 README.md                 ← You're reading this!
```

## What Each File Does (Plain English)

| File | What It Does |
|------|-------------|
| `data_collection.py` | Opens the CSV file, checks for problems (missing values, wrong types) |
| `preprocessing.py` | Guesses gender from names, creates new features, splits data 80/20 |
| `model.py` | Trains Logistic Regression, Random Forest & XGBoost, picks the best one |
| `bias_audit.py` | Checks if men and women are approved at fair rates (80% rule) |
| `visualize.py` | Creates 8 professional charts including SHAP explanations |
| `run_pipeline.py` | Runs ALL of the above in order, start to finish |
| `app.py` | Interactive web dashboard to explore results & test new loans |

## Models Used

| Model | How It Works (Simple Explanation) |
|-------|----------------------------------|
| **Logistic Regression** | Draws a line between "approve" and "deny" — simple but interpretable |
| **Random Forest** | 100 decision trees vote together — very robust and hard to fool |
| **XGBoost** | Trees learn from each other's mistakes — the most powerful model |

## Key Concepts Explained

- **Feature Engineering**: Creating new columns from existing ones (e.g., debt-to-income ratio)
- **Train/Test Split**: Training on 80% of data, testing on 20% the model has never seen
- **SHAP Values**: Explains how much each feature pushed toward approve/deny
- **Disparate Impact**: The "80% rule" — checks if approval rates are fair across groups
- **What-If Analysis**: Flipping gender to see if the model's prediction changes

## Tech Stack

Python · pandas · scikit-learn · XGBoost · SHAP · matplotlib · seaborn · Streamlit
