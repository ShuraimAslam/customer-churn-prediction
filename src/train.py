"""
Model Training & Experimentation Pipeline for Customer Churn Intelligence Dashboard.
Executes Phase 6 (Class Imbalance Mitigation), Phase 7 (Model Development),
Phase 8 (Evaluation), Phase 9 (Feature Importance), and Phase 10 (Model Persistence).
"""

import sys
from pathlib import Path

# Add project root to sys.path so direct script execution works
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from src.utils import (
    logger,
    CHURN_MODEL_PATH,
    SCALER_PATH,
    METRICS_PATH,
    FEATURE_IMPORTANCE_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    MODELS_DIR,
    save_artifact,
    save_json
)

from src.data_loader import load_raw_data
from src.preprocessing import clean_churn_data, save_cleaned_data, split_data, fit_and_apply_scaler
from src.feature_engineering import add_engineered_features, encode_features
from src.evaluate import evaluate_model_performance, build_comparison_dataframe


BUSINESS_EXPLANATIONS_MAP: Dict[str, str] = {
    "Contract_Month-to-month": "Customers on month-to-month contracts have zero switching cost or contract lock-in, creating the single largest churn catalyst.",
    "tenure": "Tenure is inversely correlated with churn. First-year customers have the highest attrition risk before becoming habituated.",
    "TotalCharges": "High cumulative billing reflects longer relationship history; lower total charges typically mark early-tenure flight risks.",
    "MonthlyCharges": "High monthly bills without perceived premium value generate pricing friction, prompting subscribers to seek cheaper alternatives.",
    "AverageMonthlySpend": "High burn rate relative to tenure signals bill volatility or recent rate hikes that accelerate departure.",
    "InternetService_Fiber optic": "Fiber optic subscribers face steep monthly fees and frequent service quality expectations; any downtime triggers rapid switching.",
    "PaymentMethod_Electronic check": "Electronic check users lack the recurring automation of credit cards and bank transfers, driving higher voluntary and involuntary dropouts.",
    "TechSupport_No": "Subscribers without technical support struggle to resolve connectivity issues independently, resulting in customer dissatisfaction.",
    "OnlineSecurity_No": "Absence of cybersecurity protection leaves subscribers vulnerable to phishing/threats and lowers their perceived service value.",
    "ServiceCount": "Low total service adoption minimizes lock-in, enabling effortless migration to competing carriers.",
    "PaperlessBilling_Yes": "Paperless billing users check invoices digitally and are more inclined to audit line items and shop competing digital offerings.",
    "SeniorCitizen": "Senior citizens often have fixed incomes or distinct service requirements that demand specialized retention packages.",
    "OnlineBackup_No": "Lack of cloud data backup lowers emotional switching friction.",
    "DeviceProtection_No": "Without equipment replacement plans, hardware damages prompt customers to switch carriers for new promotional phone deals."
}


def run_training_pipeline() -> Dict[str, Any]:
    """
    Executes the end-to-end training, benchmarking, and persistence pipeline.
    
    Returns:
        Dictionary containing benchmark summary, best model name, and metric metadata.
    """
    logger.info("=" * 60)
    logger.info("STARTING CUSTOMER CHURN ML PIPELINE")
    logger.info("=" * 60)

    # 1. Load Raw Data
    df_raw = load_raw_data()
    
    # 2. Clean Data
    df_cleaned, quality_report = clean_churn_data(df_raw)
    save_cleaned_data(df_cleaned)

    # 3. Feature Engineering
    logger.info("Engineering business features (TenureCategory, Spend Ratio, Bundles)...")
    df_features = add_engineered_features(df_cleaned)

    # 4. Stratified Train/Test Split (80/20)
    X_train_raw, X_test_raw, y_train, y_test = split_data(
        df_features, target_col="Churn", test_size=0.20, random_state=42
    )

    # 5. One-Hot Encoding
    logger.info("Applying One-Hot Encoding with column consistency alignment...")
    X_train_encoded, feature_columns = encode_features(X_train_raw, is_training=True)
    X_test_encoded, _ = encode_features(X_test_raw, is_training=False, expected_columns=feature_columns)

    # 6. Scaling Numerical Features
    numerical_cols = [
        col for col in ["tenure", "MonthlyCharges", "TotalCharges", "AverageMonthlySpend", "ServiceCount"]
        if col in X_train_encoded.columns
    ]
    X_train_scaled, X_test_scaled, scaler = fit_and_apply_scaler(
        X_train_encoded, X_test_encoded, numerical_cols, scaler_path=SCALER_PATH
    )

    # 7. Imbalance Handling Analysis (Class Weighting vs SMOTE)
    imbalance_stats = {
        "original_train_distribution": y_train.value_counts().to_dict(),
        "original_train_ratio": {
            "Retained (0)": f"{(y_train == 0).mean() * 100:.1f}%",
            "Churned (1)": f"{(y_train == 1).mean() * 100:.1f}%"
        }
    }
    logger.info(f"Original Training Target Ratio: {imbalance_stats['original_train_ratio']}")

    # Apply SMOTE for benchmarking
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    imbalance_stats["smote_train_distribution"] = y_train_smote.value_counts().to_dict()
    logger.info(f"SMOTE Balanced Distribution: {imbalance_stats['smote_train_distribution']}")

    # 8. Model Training & Evaluation Suite
    models_to_evaluate = {
        "Logistic Regression (Class Weighted)": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Logistic Regression (SMOTE)": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        ),
        "Random Forest (Class Weighted)": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest (SMOTE)": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    }

    evaluation_results = []
    trained_models = {}

    for name, model in models_to_evaluate.items():
        logger.info(f"Training [{name}]...")
        if "SMOTE" in name:
            model.fit(X_train_smote, y_train_smote)
        else:
            model.fit(X_train_scaled, y_train)

        res = evaluate_model_performance(model, X_test_scaled, y_test, model_name=name)
        evaluation_results.append(res)
        trained_models[name] = model

    comparison_df = build_comparison_dataframe(evaluation_results)
    logger.info("\n" + comparison_df.to_string(index=False))

    # 9. Best Model Selection
    # For churn prediction in business, we seek maximum ROC-AUC and high recall (catching leavers)
    # without sacrificing precision below acceptable bounds.
    best_result = max(evaluation_results, key=lambda x: (x["metrics"]["roc_auc"], x["metrics"]["f1_score"]))
    best_model_name = best_result["model_name"]
    best_model = trained_models[best_model_name]
    logger.info(f"Selected Best Production Model: [{best_model_name}] with ROC-AUC = {best_result['metrics']['roc_auc']}")

    # 10. Extract Feature Importance
    logger.info("Computing Feature Importances & Business Interpretations...")
    feature_importance_list = []
    
    if hasattr(best_model, "feature_importances_"):
        raw_importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        raw_importances = np.abs(best_model.coef_[0])
    else:
        # Fallback to RF
        rf_model = trained_models["Random Forest (Class Weighted)"]
        raw_importances = rf_model.feature_importances_

    sorted_idx = np.argsort(raw_importances)[::-1]
    for idx in sorted_idx[:15]:
        feat = feature_columns[idx]
        imp = float(raw_importances[idx])
        explanation = BUSINESS_EXPLANATIONS_MAP.get(
            feat,
            f"The feature '{feat}' has a direct proportional influence on subscription longevity and churn probability."
        )
        feature_importance_list.append({
            "feature": feat,
            "importance": round(imp, 4),
            "percentage": round(imp / sum(raw_importances) * 100, 2),
            "business_explanation": explanation
        })

    # 11. Model Persistence (Phase 10)
    save_artifact(best_model, CHURN_MODEL_PATH)
    # Also save dedicated models for direct access
    rf_best = trained_models["Random Forest (Class Weighted)"]
    lr_best = trained_models["Logistic Regression (Class Weighted)"]
    save_artifact(rf_best, MODELS_DIR / "random_forest_model.pkl")
    save_artifact(lr_best, MODELS_DIR / "logistic_regression_model.pkl")

    # Save train and test splits to data/processed
    train_combined = X_train_scaled.copy()
    train_combined["Churn"] = y_train.values
    train_combined.to_csv(TRAIN_DATA_PATH, index=False)
    
    test_combined = X_test_scaled.copy()
    test_combined["Churn"] = y_test.values
    test_combined.to_csv(TEST_DATA_PATH, index=False)
    logger.info(f"Saved processed train.csv ({len(train_combined)} rows) and test.csv ({len(test_combined)} rows)")

    save_json({
        "best_model_name": best_model_name,
        "imbalance_handling": "Class Weighting vs SMOTE benchmarked",
        "best_metrics": best_result["metrics"],
        "all_models": evaluation_results,
        "quality_report": quality_report,
        "imbalance_stats": imbalance_stats
    }, METRICS_PATH)
    
    save_json({
        "model_used": best_model_name,
        "top_features": feature_importance_list
    }, FEATURE_IMPORTANCE_PATH)

    logger.info("All model artifacts, scalers, and metric reports safely persisted!")
    logger.info("=" * 60)

    return {
        "best_model_name": best_model_name,
        "best_metrics": best_result["metrics"],
        "comparison_table": comparison_df.to_dict(orient="records"),
        "top_features": feature_importance_list[:10]
    }


if __name__ == "__main__":
    run_training_pipeline()
