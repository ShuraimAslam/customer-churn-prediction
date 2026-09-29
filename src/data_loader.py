"""
Data Loader Module for Customer Churn Intelligence Dashboard.
Handles loading raw data, schema verification, data dictionary generation,
and comprehensive Phase 1 data understanding reporting.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from src.utils import logger, RAW_DATA_PATH


FEATURE_DICTIONARY: Dict[str, Dict[str, str]] = {
    "customerID": {
        "type": "Categorical (ID)",
        "description": "Unique alphanumeric customer identifier",
        "business_relevance": "Used for tracking individual customer records; dropped before modeling."
    },
    "gender": {
        "type": "Categorical (Binary)",
        "description": "Customer gender (Male, Female)",
        "business_relevance": "Demographic factor to verify bias and service parity."
    },
    "SeniorCitizen": {
        "type": "Categorical (Binary: 0/1)",
        "description": "Whether the customer is a senior citizen (1) or not (0)",
        "business_relevance": "Seniors often have distinct service, support, and billing preferences."
    },
    "Partner": {
        "type": "Categorical (Binary: Yes/No)",
        "description": "Whether the customer has a partner/spouse",
        "business_relevance": "Couples or families often have higher service stickiness than single subscribers."
    },
    "Dependents": {
        "type": "Categorical (Binary: Yes/No)",
        "description": "Whether the customer lives with dependents/children",
        "business_relevance": "Households with dependents frequently depend on uninterrupted connectivity."
    },
    "tenure": {
        "type": "Numerical (Integer: Months)",
        "description": "Number of months the customer has stayed with the company",
        "business_relevance": "Primary indicator of customer lifetime and maturity. Higher tenure typically lowers churn."
    },
    "PhoneService": {
        "type": "Categorical (Binary: Yes/No)",
        "description": "Whether the customer subscribes to landline telephone service",
        "business_relevance": "Core legacy telecom product."
    },
    "MultipleLines": {
        "type": "Categorical (Yes/No/No phone service)",
        "description": "Whether the customer has multiple telephone lines",
        "business_relevance": "Multi-line accounts represent deeper commitment and higher ARPU."
    },
    "InternetService": {
        "type": "Categorical (DSL/Fiber optic/No)",
        "description": "Customer's internet service provider technology",
        "business_relevance": "Fiber optic provides high speed but historically exhibits high churn due to pricing/expectations."
    },
    "OnlineSecurity": {
        "type": "Categorical (Yes/No/No internet service)",
        "description": "Whether the customer has online security add-on",
        "business_relevance": "Value-add security services boost customer lock-in and trust."
    },
    "OnlineBackup": {
        "type": "Categorical (Yes/No/No internet service)",
        "description": "Whether the customer has cloud backup add-on",
        "business_relevance": "Data backup increases switching costs for the subscriber."
    },
    "DeviceProtection": {
        "type": "Categorical (Yes/No/No internet service)",
        "description": "Whether customer has hardware insurance / device warranty",
        "business_relevance": "Protective insurance lowers dissatisfaction from broken equipment."
    },
    "TechSupport": {
        "type": "Categorical (Yes/No/No internet service)",
        "description": "Whether the customer has priority technical support service",
        "business_relevance": "Critical churn mitigating service; users without tech support churn faster when facing issues."
    },
    "StreamingTV": {
        "type": "Categorical (Yes/No/No internet service)",
        "description": "Whether customer subscribes to streaming TV through provider",
        "business_relevance": "Content bundling deepens daily engagement."
    },
    "StreamingMovies": {
        "type": "Categorical (Yes/No/No internet service)",
        "description": "Whether customer subscribes to movie streaming through provider",
        "business_relevance": "Entertainment bundle indicator."
    },
    "Contract": {
        "type": "Categorical (Month-to-month / One year / Two year)",
        "description": "Customer's contract commitment term",
        "business_relevance": "Most critical churn predictor. Month-to-month contracts allow friction-free switching."
    },
    "PaperlessBilling": {
        "type": "Categorical (Yes/No)",
        "description": "Whether the customer opts for electronic e-billing",
        "business_relevance": "Digital billing correlation with tech-savviness and payment habits."
    },
    "PaymentMethod": {
        "type": "Categorical (Electronic check / Mailed check / Bank transfer / Credit card)",
        "description": "Customer's primary billing payment method",
        "business_relevance": "Electronic checks frequently suffer from payment failures and higher voluntary/involuntary churn."
    },
    "MonthlyCharges": {
        "type": "Numerical (Float: USD)",
        "description": "The amount charged to the customer monthly",
        "business_relevance": "Key pricing metric; high monthly bills without perceived value trigger churn."
    },
    "TotalCharges": {
        "type": "Numerical (Float: USD stored as string in raw)",
        "description": "The cumulative amount charged to the customer over tenure",
        "business_relevance": "Lifetime monetary value; needs conversion from string and handling for 0-tenure blanks."
    },
    "Churn": {
        "type": "Target (Binary: Yes/No)",
        "description": "Whether the customer left the company within the last month",
        "business_relevance": "Primary target variable for classification. Yes=1, No=0."
    }
}


def load_raw_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads the raw Telco customer churn dataset from CSV.
    
    Args:
        file_path: Optional custom file path. Defaults to RAW_DATA_PATH.
        
    Returns:
        pd.DataFrame containing the raw dataset.
    """
    path = file_path or RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {path}")
    
    logger.info(f"Loading raw dataset from {path}")
    df = pd.read_csv(path)
    logger.info(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def get_feature_dictionary() -> Dict[str, Dict[str, str]]:
    """Returns the comprehensive business feature dictionary."""
    return FEATURE_DICTIONARY


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extracts Phase 1 Data Understanding metrics:
    - Shape (rows, columns)
    - Column data types
    - Missing value counts and percentages
    - Target variable distribution
    - Summary statistics
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing structured summary metrics.
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    
    # Check for whitespace-only blanks (known issue in TotalCharges)
    blank_spaces = {}
    for col in df.columns:
        if df[col].dtype == "object":
            blanks = (df[col].astype(str).str.strip() == "").sum()
            if blanks > 0:
                blank_spaces[col] = int(blanks)

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    
    summary = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "missing_values": {col: int(missing_count[col]) for col in df.columns if missing_count[col] > 0},
        "blank_space_values": blank_spaces,
        "target_distribution": df["Churn"].value_counts(dropna=False).to_dict() if "Churn" in df.columns else {},
        "target_percentage": (df["Churn"].value_counts(normalize=True) * 100).round(2).to_dict() if "Churn" in df.columns else {}
    }
    return summary


def generate_data_understanding_report(df: pd.DataFrame) -> str:
    """
    Generates a professional Markdown report for Phase 1 Data Understanding.
    
    Args:
        df: Input raw DataFrame
        
    Returns:
        Markdown-formatted report string.
    """
    summary = get_dataset_summary(df)
    rows = summary["shape"]["rows"]
    cols = summary["shape"]["columns"]
    
    report = [
        "# Phase 1: Data Understanding Report",
        "",
        "## 1. Dataset Dimensions",
        f"- **Total Customer Records (Rows):** `{rows:,}`",
        f"- **Total Attributes (Columns):** `{cols}`",
        f"- **Target Variable:** `Churn` (Values: `Yes`, `No`)",
        "",
        "## 2. Feature Classification",
        f"- **Numerical Attributes ({len(summary['numerical_columns'])}):** {', '.join(summary['numerical_columns'])}",
        f"- **Categorical Attributes ({len(summary['categorical_columns'])}):** {', '.join(summary['categorical_columns'])}",
        "",
        "## 3. Data Hygiene & Missing Value Analysis",
    ]
    
    if summary["blank_space_values"]:
        for col, count in summary["blank_space_values"].items():
            report.append(f"- **Whitespace String Anomaly Detected:** Column `{col}` contains `{count}` blank strings (`' '`). These correspond to customers with tenure = 0 who have not accrued a full billing cycle.")
    else:
        report.append("- No explicit NaN values or whitespace anomalies detected in raw inspection.")
        
    if "target_distribution" in summary and summary["target_distribution"]:
        churn_dist = summary["target_distribution"]
        churn_pct = summary["target_percentage"]
        report.extend([
            "",
            "## 4. Class Distribution (Target Imbalance)",
            f"- **Retained Customers (No):** `{churn_dist.get('No', 0):,}` ({churn_pct.get('No', 0.0)}%)",
            f"- **Churned Customers (Yes):** `{churn_dist.get('Yes', 0):,}` ({churn_pct.get('Yes', 0.0)}%)",
            "- **Business Note:** The dataset reflects a ~73.5% retention to ~26.5% churn distribution. This moderate class imbalance requires careful handling via stratified splits, class weighting or SMOTE, and recall-focused metric optimization.",
        ])
        
    return "\n".join(report)


if __name__ == "__main__":
    df_raw = load_raw_data()
    print(generate_data_understanding_report(df_raw))
