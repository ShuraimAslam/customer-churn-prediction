"""
Evaluation Module for Customer Churn Intelligence Dashboard.
Computes multi-dimensional classification metrics, confusion matrices,
ROC curves, precision-recall metrics, and model comparison leaderboards.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
from src.utils import logger


def evaluate_model_performance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier across all standard business and ML metrics.
    
    Args:
        model: Trained classifier supporting predict and predict_proba
        X_test: Test features
        y_test: True ground truth labels
        model_name: Display name for the model
        
    Returns:
        Dictionary containing scalar metrics, confusion matrix, and curve coordinates.
    """
    y_pred = model.predict(X_test)
    
    # Check if model supports probability estimation
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.5
    
    # Compute confusion matrix [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    
    # ROC curve points
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    
    # Precision-Recall curve points
    prec_pts, rec_pts, pr_thresholds = precision_recall_curve(y_test, y_prob)
    
    # Downsample curve points for compact JSON storage
    stride_roc = max(1, len(fpr) // 50)
    stride_pr = max(1, len(prec_pts) // 50)
    
    results = {
        "model_name": model_name,
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4)
        },
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "total_tested": int(len(y_test))
        },
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr[::stride_roc]],
            "tpr": [round(float(y), 4) for y in tpr[::stride_roc]]
        },
        "pr_curve": {
            "precision": [round(float(p), 4) for p in prec_pts[::stride_pr]],
            "recall": [round(float(r), 4) for r in rec_pts[::stride_pr]]
        },
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    logger.info(
        f"[{model_name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}"
    )
    return results


def build_comparison_dataframe(model_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Builds a structured comparison DataFrame from list of model evaluation results.
    
    Args:
        model_results: List of evaluation result dictionaries
        
    Returns:
        Formatted pd.DataFrame
    """
    records = []
    for res in model_results:
        m = res["metrics"]
        cm = res["confusion_matrix"]
        records.append({
            "Model": res["model_name"],
            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
            "Precision": f"{m['precision'] * 100:.2f}%",
            "Recall (Churn)": f"{m['recall'] * 100:.2f}%",
            "F1-Score": f"{m['f1_score']:.4f}",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "True Positives (Caught)": cm["true_positives"],
            "False Negatives (Missed)": cm["false_negatives"]
        })
    return pd.DataFrame(records)
