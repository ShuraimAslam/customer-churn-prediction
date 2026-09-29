"""
Feature Engineering Module for Customer Churn Intelligence Dashboard.
Implements domain-specific feature creation, tenure binning, value tiering,
one-hot encoding, and column alignment for robust production inference.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

from src.utils import logger, FEATURES_PATH, save_json, load_json


FEATURE_RATIONALE = {
    "AverageMonthlySpend": {
        "formula": "TotalCharges / (tenure + 1)",
        "why_it_exists": "Captures the historical effective burn rate of the customer. A customer with a high monthly spend ratio compared to their current monthly charges may indicate bill increases or inconsistent promotional periods.",
        "expected_impact": "High values indicate recent bill shock or heavy usage, positively correlated with churn risk."
    },
    "TenureCategory": {
        "formula": "Binned tenure: 0-12m (New), 13-24m (Bronze), 25-48m (Silver), 49-72m (Gold)",
        "why_it_exists": "Discretizes customer relationship life-cycles into distinct behavioral cohorts. The onboarding window (0-12m) is the most volatile period for telecoms.",
        "expected_impact": "New cohort has substantially higher churn risk (>40%), whereas Gold cohort exhibits high loyalty (<10% churn)."
    },
    "HighValueCustomer": {
        "formula": "1 if (MonthlyCharges >= 70.0 and tenure >= 12) else 0",
        "why_it_exists": "Identifies long-term high ARPU (Average Revenue Per User) accounts that represent maximum revenue vulnerability.",
        "expected_impact": "Differentiates high-margin enterprise/premium subscribers from low-cost budget accounts."
    },
    "ServiceCount": {
        "formula": "Sum of subscribed value-added services (Security, Backup, Protection, Support, TV, Movies, Phone)",
        "why_it_exists": "Quantifies ecosystem lock-in. Customers with multiple integrated services encounter friction and high switching costs to competitors.",
        "expected_impact": "Negative correlation with churn: more services dramatically lower churn risk."
    },
    "Contract_Risk_Flag": {
        "formula": "1 if Contract == 'Month-to-month' else 0",
        "why_it_exists": "Month-to-month contracts lack long-term contractual switching friction, enabling spontaneous departures.",
        "expected_impact": "Primary positive driver of churn in telecommunications."
    },
    "HasSupportAndSecurity": {
        "formula": "1 if (OnlineSecurity == 'Yes' and TechSupport == 'Yes') else 0",
        "why_it_exists": "Security and technical support address the two primary customer frustration points (security concerns and network issues).",
        "expected_impact": "Significantly decreases churn by ensuring smooth service reliability."
    }
}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes business features:
    - AverageMonthlySpend
    - TenureCategory
    - HighValueCustomer
    - ServiceCount
    - Contract_Risk_Flag
    - HasSupportAndSecurity
    
    Args:
        df: Cleaned DataFrame (with tenure, MonthlyCharges, TotalCharges, etc.)
        
    Returns:
        DataFrame augmented with engineered features.
    """
    df_feat = df.copy()
    
    # 1. Average Monthly Spend (smooth tenure with +1 to prevent div by 0)
    tenure_safe = df_feat["tenure"] if "tenure" in df_feat.columns else 0
    if "TotalCharges" in df_feat.columns:
        df_feat["AverageMonthlySpend"] = (df_feat["TotalCharges"] / (tenure_safe + 1)).round(2)
    else:
        df_feat["AverageMonthlySpend"] = df_feat.get("MonthlyCharges", 0.0)

    # 2. Tenure Category Bins
    if "tenure" in df_feat.columns:
        bins = [-1, 12, 24, 48, 100]
        labels = ["0-12m (New)", "13-24m (Bronze)", "25-48m (Silver)", "49-72m (Gold)"]
        df_feat["TenureCategory"] = pd.cut(df_feat["tenure"], bins=bins, labels=labels).astype(str)

    # 3. High Value Customer Flag (MonthlyCharges >= 70 & tenure >= 12)
    if "MonthlyCharges" in df_feat.columns and "tenure" in df_feat.columns:
        df_feat["HighValueCustomer"] = (
            (df_feat["MonthlyCharges"] >= 70.0) & (df_feat["tenure"] >= 12)
        ).astype(int)
    else:
        df_feat["HighValueCustomer"] = 0

    # 4. Service Count (Quantifies sticky ecosystem usage)
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    service_sum = pd.Series(0, index=df_feat.index)
    for col in service_cols:
        if col in df_feat.columns:
            service_sum += (df_feat[col] == "Yes").astype(int)
    if "PhoneService" in df_feat.columns:
        service_sum += (df_feat["PhoneService"] == "Yes").astype(int)
    if "InternetService" in df_feat.columns:
        service_sum += (df_feat["InternetService"].isin(["DSL", "Fiber optic"])).astype(int)
    df_feat["ServiceCount"] = service_sum

    # 5. Contract Risk Flag (Month-to-month contracts have highest volatility)
    if "Contract" in df_feat.columns:
        df_feat["Contract_Risk_Flag"] = (df_feat["Contract"] == "Month-to-month").astype(int)
        
    # 6. Technical Peace-of-Mind (Security + TechSupport)
    if "OnlineSecurity" in df_feat.columns and "TechSupport" in df_feat.columns:
        df_feat["HasSupportAndSecurity"] = (
            (df_feat["OnlineSecurity"] == "Yes") & (df_feat["TechSupport"] == "Yes")
        ).astype(int)

    return df_feat


def encode_features(
    df: pd.DataFrame,
    is_training: bool = True,
    expected_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Applies One-Hot Encoding to categorical variables, ensuring consistent
    feature alignment between training and inference phases.
    
    Args:
        df: Input feature DataFrame
        is_training: If True, saves feature column order to models/feature_names.json
        expected_columns: List of expected columns for inference alignment
        
    Returns:
        Tuple of (encoded_df, feature_column_names)
    """
    df_encoded = df.copy()
    
    # Drop customer identifier if present
    if "customerID" in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=["customerID"])
        
    # Identify categorical columns to one-hot encode
    cat_cols = df_encoded.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Exclude target if present in dataframe
    if "Churn" in cat_cols:
        cat_cols.remove("Churn")

    if is_training:
        # One-hot encode with drop_first=True to avoid multicollinearity during model fitting
        df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True, dtype=int)
        feature_columns = [c for c in df_encoded.columns if c != "Churn"]
        save_json({"features": feature_columns}, FEATURES_PATH)
        logger.info(f"Saved {len(feature_columns)} feature names to {FEATURES_PATH}")
        return df_encoded, feature_columns
    else:
        # For inference (single row or batch), use drop_first=False so single-row records
        # do not have their only category dropped, then align strictly to expected training columns.
        df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=False, dtype=int)
        if expected_columns is None:
            expected_columns = load_json(FEATURES_PATH)["features"]
            
        aligned_df = align_features(df_encoded, expected_columns)
        return aligned_df, expected_columns


def align_features(df: pd.DataFrame, expected_columns: List[str]) -> pd.DataFrame:
    """
    Aligns inference features to the exact column set and order produced during training.
    Missing dummy columns are filled with 0; extraneous columns are dropped.
    
    Args:
        df: Encoded inference DataFrame
        expected_columns: List of feature names from training
        
    Returns:
        Aligned DataFrame matching expected_columns exactly.
    """
    aligned_df = pd.DataFrame(0, index=df.index, columns=expected_columns)
    
    for col in expected_columns:
        if col in df.columns:
            aligned_df[col] = df[col]
            
    return aligned_df


def get_feature_explanations() -> Dict[str, Dict[str, str]]:
    """Returns the rationale dictionary for all engineered features."""
    return FEATURE_RATIONALE
