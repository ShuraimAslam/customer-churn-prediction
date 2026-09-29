"""
Data Preprocessing Module for Customer Churn Intelligence Dashboard.
Handles data cleaning, type conversion, duplicate removal, quality validation,
train-test splitting, scaling, and preprocessing artifact persistence.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import (
    logger,
    CLEANED_DATA_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    SCALER_PATH,
    save_artifact,
    load_artifact,
)


def clean_churn_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw Telco Customer Churn data:
    1. Strips column names and strings.
    2. Converts 'TotalCharges' from string/object to float64.
       Replaces blank spaces (' ') with 0.0 (tenure=0 customers who haven't billed yet).
    3. Verifies and drops duplicate rows if any exist.
    4. Encodes target 'Churn' as binary: 'Yes' -> 1, 'No' -> 0 (if present).
    5. Normalizes categorical string values.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Tuple of (cleaned_df, quality_report_dict)
    """
    logger.info("Initiating Phase 2: Data Cleaning & Hygiene pipeline...")
    initial_rows, initial_cols = df.shape
    initial_missing = int(df.isnull().sum().sum())
    
    cleaned_df = df.copy()
    
    # 1. Strip whitespace from column headers
    cleaned_df.columns = cleaned_df.columns.str.strip()
    
    # 2. Check for duplicate customer records
    duplicate_count = int(cleaned_df.duplicated().sum())
    if "customerID" in cleaned_df.columns:
        duplicate_ids = int(cleaned_df["customerID"].duplicated().sum())
    else:
        duplicate_ids = 0
        
    if duplicate_count > 0:
        cleaned_df = cleaned_df.drop_duplicates()
        logger.info(f"Dropped {duplicate_count} duplicate rows.")

    # 3. Clean 'TotalCharges'
    blanks_in_total_charges = 0
    if "TotalCharges" in cleaned_df.columns:
        if cleaned_df["TotalCharges"].dtype == "object":
            blanks_mask = cleaned_df["TotalCharges"].astype(str).str.strip() == ""
            blanks_in_total_charges = int(blanks_mask.sum())
            logger.info(f"Found {blanks_in_total_charges} blank entries in TotalCharges (tenure=0). Imputing with 0.0.")
            cleaned_df["TotalCharges"] = pd.to_numeric(cleaned_df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce")
            # Fill with 0.0 as tenure is 0 for these accounts
            cleaned_df["TotalCharges"] = cleaned_df["TotalCharges"].fillna(0.0)
        else:
            cleaned_df["TotalCharges"] = cleaned_df["TotalCharges"].fillna(0.0)

    # 4. Ensure SeniorCitizen is integer
    if "SeniorCitizen" in cleaned_df.columns:
        cleaned_df["SeniorCitizen"] = cleaned_df["SeniorCitizen"].astype(int)

    # 5. Clean target column if present
    churn_mapping = {}
    if "Churn" in cleaned_df.columns:
        churn_mapping = {"No": 0, "Yes": 1}
        if cleaned_df["Churn"].dtype == "object":
            cleaned_df["Churn"] = cleaned_df["Churn"].map(churn_mapping).astype(int)

    # 6. Verify final missing values
    final_missing = int(cleaned_df.isnull().sum().sum())
    final_rows, final_cols = cleaned_df.shape
    
    quality_report = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "initial_cols": initial_cols,
        "final_cols": final_cols,
        "duplicates_removed": duplicate_count,
        "duplicate_customer_ids": duplicate_ids,
        "blanks_imputed_total_charges": blanks_in_total_charges,
        "initial_missing_cells": initial_missing,
        "final_missing_cells": final_missing,
        "data_quality_score": 100.0 if final_missing == 0 else round((1 - final_missing / (final_rows * final_cols)) * 100, 2)
    }
    
    logger.info(f"Data cleaning completed. Quality score: {quality_report['data_quality_score']}%")
    return cleaned_df, quality_report


def save_cleaned_data(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Saves cleaned DataFrame to CSV."""
    path = output_path or CLEANED_DATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Cleaned dataset saved to: {path}")
    return path


def split_data(
    df: pd.DataFrame,
    target_col: str = "Churn",
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified 80/20 train/test split.
    
    Args:
        df: Processed DataFrame
        target_col: Name of target column
        test_size: Ratio of test partition (default 0.20)
        random_state: Reproducibility seed (default 42)
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
        
    X = df.drop(columns=[target_col])
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])
        
    y = df[target_col]
    
    logger.info(f"Splitting dataset: test_size={test_size}, stratify on '{target_col}', random_state={random_state}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    logger.info(f"Train split shape: {X_train.shape} | Test split shape: {X_test.shape}")
    logger.info(f"Train target balance: {y_train.value_counts(normalize=True).round(3).to_dict()}")
    return X_train, X_test, y_train, y_test


def fit_and_apply_scaler(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numerical_cols: List[str],
    scaler_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fits StandardScaler on numerical features of training data and transforms both train and test.
    Saves the fitted scaler artifact.
    
    Args:
        X_train: Training features DataFrame
        X_test: Testing features DataFrame
        numerical_cols: List of numerical column names to scale
        scaler_path: Path to save scaler artifact
        
    Returns:
        Scaled X_train, Scaled X_test, fitted StandardScaler
    """
    scaler = StandardScaler()
    
    valid_cols = [c for c in numerical_cols if c in X_train.columns]
    logger.info(f"Scaling numerical columns: {valid_cols}")
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[valid_cols] = scaler.fit_transform(X_train[valid_cols])
    X_test_scaled[valid_cols] = scaler.transform(X_test[valid_cols])
    
    save_path = scaler_path or SCALER_PATH
    save_artifact(scaler, save_path)
    
    return X_train_scaled, X_test_scaled, scaler


def load_scaler(scaler_path: Optional[Path] = None) -> StandardScaler:
    """Loads saved StandardScaler artifact."""
    path = scaler_path or SCALER_PATH
    return load_artifact(path)
