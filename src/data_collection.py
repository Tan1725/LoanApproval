# ============================================================
# 📁 data_collection.py — Step 1: Loading Your Raw Data
# ============================================================
# 
# WHAT THIS FILE DOES:
# --------------------
# This is the very first step in any ML project: getting your data.
# 
# In real-world projects, data might come from databases, APIs, 
# web scraping, etc. In our case, we have a CSV file (like an 
# Excel spreadsheet) that we need to load and validate.
#
# Think of this file as the "delivery truck" — it brings the 
# raw ingredients (data) into our kitchen (project) so we can
# start cooking (building ML models).
#
# WHAT YOU'LL LEARN:
# ------------------
# - How to load CSV files using pandas
# - How to do basic data validation (checking for problems)
# - How to get a quick summary of your data
# ============================================================

import pandas as pd  # pandas = the #1 library for working with tabular data
import os            # os = helps us work with file paths and directories


def load_data(filepath):
    """
    Load the loan approval CSV file and return it as a DataFrame.
    
    A DataFrame is basically a spreadsheet in Python — it has rows 
    and columns, just like Excel. Each row is one loan application,
    each column is one piece of information about that application.
    
    Parameters:
    -----------
    filepath : str
        The path to the CSV file (e.g., "loan_approval.csv")
    
    Returns:
    --------
    df : pandas DataFrame
        The loaded data, ready for analysis
    """
    
    # --- Step 1: Check if the file actually exists ---
    # Before trying to open a file, always check it's there!
    # This prevents confusing error messages later.
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"❌ Could not find the file: {filepath}\n"
            f"   Make sure the CSV file is in the right folder!"
        )
    
    # --- Step 2: Load the CSV file into a DataFrame ---
    # pd.read_csv() reads the CSV and turns it into a table (DataFrame)
    # Think of it like opening an Excel file, but in Python.
    print(f"📂 Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    
    # --- Step 3: Print a quick summary so we know what we're working with ---
    print(f"✅ Data loaded successfully!")
    print(f"   📊 Total rows (loan applications): {len(df)}")
    print(f"   📋 Total columns (features): {len(df.columns)}")
    print(f"   📝 Column names: {list(df.columns)}")
    
    return df


def validate_data(df):
    """
    Check the data for common problems before we start working with it.
    
    This is like a quality check in a factory — before we start building
    something, we make sure our raw materials aren't damaged.
    
    We check for:
    1. Missing values (empty cells)
    2. Correct data types (numbers should be numbers, not text)
    3. Reasonable value ranges (no negative ages, etc.)
    
    Parameters:
    -----------
    df : pandas DataFrame
        The raw data to validate
    
    Returns:
    --------
    report : dict
        A dictionary containing the validation results
    """
    
    print("\n🔍 Validating data quality...")
    print("=" * 50)
    
    report = {}
    
    # --- Check 1: Missing values ---
    # Missing values are like blank cells in Excel.
    # They can cause problems for ML models, so we need to know about them.
    missing = df.isnull().sum()  # Count missing values per column
    total_missing = missing.sum()
    report['missing_values'] = missing.to_dict()
    
    if total_missing == 0:
        print("✅ No missing values found — perfect!")
    else:
        print(f"⚠️  Found {total_missing} missing values:")
        for col, count in missing.items():
            if count > 0:
                print(f"   - {col}: {count} missing ({count/len(df)*100:.1f}%)")
    
    # --- Check 2: Data types ---
    # Each column should have the right type:
    #   - Numbers (int/float) for things like income, credit score
    #   - Text (object) for things like names, cities
    #   - Boolean (True/False) for loan_approved
    print(f"\n📋 Column types:")
    for col in df.columns:
        dtype = df[col].dtype
        print(f"   - {col}: {dtype}")
    report['dtypes'] = df.dtypes.astype(str).to_dict()
    
    # --- Check 3: Basic statistics ---
    # This gives us min, max, mean, etc. for each numeric column.
    # Useful for spotting weird values (like negative income).
    print(f"\n📊 Quick statistics for numeric columns:")
    numeric_stats = df.describe()
    print(numeric_stats.to_string())
    report['statistics'] = numeric_stats.to_dict()
    
    # --- Check 4: Target variable distribution ---
    # The "target" is what we're trying to predict: loan_approved (True/False)
    # We want to know: how many approvals vs denials?
    # If it's very unbalanced (e.g., 99% approved), that's important to know.
    if 'loan_approved' in df.columns:
        approval_counts = df['loan_approved'].value_counts()
        total = len(df)
        print(f"\n🎯 Target variable (loan_approved) distribution:")
        for value, count in approval_counts.items():
            print(f"   - {value}: {count} ({count/total*100:.1f}%)")
        report['target_distribution'] = approval_counts.to_dict()
    
    print("=" * 50)
    print("✅ Validation complete!\n")
    
    return report


def get_data_summary(df):
    """
    Create a human-readable summary of the dataset.
    
    This is useful for presentations and reports — it gives you
    the "big picture" of what the data looks like.
    
    Parameters:
    -----------
    df : pandas DataFrame
    
    Returns:
    --------
    summary : dict
        Key statistics about the dataset
    """
    
    summary = {
        'total_applications': len(df),
        'total_features': len(df.columns),
        'columns': list(df.columns),
        'avg_income': df['income'].mean() if 'income' in df.columns else None,
        'avg_credit_score': df['credit_score'].mean() if 'credit_score' in df.columns else None,
        'avg_loan_amount': df['loan_amount'].mean() if 'loan_amount' in df.columns else None,
        'approval_rate': df['loan_approved'].mean() * 100 if 'loan_approved' in df.columns else None,
    }
    
    print("📋 Dataset Summary:")
    print(f"   Total applications: {summary['total_applications']}")
    if summary['avg_income']:
        print(f"   Average income: ${summary['avg_income']:,.0f}")
    if summary['avg_credit_score']:
        print(f"   Average credit score: {summary['avg_credit_score']:.0f}")
    if summary['avg_loan_amount']:
        print(f"   Average loan amount: ${summary['avg_loan_amount']:,.0f}")
    if summary['approval_rate']:
        print(f"   Overall approval rate: {summary['approval_rate']:.1f}%")
    
    return summary


# ============================================================
# This block runs ONLY when you execute this file directly:
#   python src/data_collection.py
# It does NOT run when another file imports from this file.
# ============================================================
if __name__ == "__main__":
    # Quick test — load and validate the data
    df = load_data("loan_approval.csv")
    validate_data(df)
    get_data_summary(df)
