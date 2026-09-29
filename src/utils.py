"""
Utility functions and system constants for Customer Churn Intelligence Dashboard.
Provides path management, artifact persistence, metric formatting, and styling helpers.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib

# Setup standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("ChurnIntelligence")

# Project Root Directory Resolution
BASE_DIR = Path(__file__).resolve().parent.parent

# Core Paths
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
APP_DIR = BASE_DIR / "app"

# Standard Data File Paths
RAW_DATA_PATH = DATA_RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEANED_DATA_PATH = DATA_PROCESSED_DIR / "cleaned_churn_data.csv"
TRAIN_DATA_PATH = DATA_PROCESSED_DIR / "train.csv"
TEST_DATA_PATH = DATA_PROCESSED_DIR / "test.csv"

# Model Artifact Paths
CHURN_MODEL_PATH = MODELS_DIR / "churn_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURES_PATH = MODELS_DIR / "feature_names.json"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.json"

# Risk Categories Configuration
RISK_CATEGORIES = {
    "Low": {
        "range": (0.0, 0.30),
        "label": "Low Risk",
        "color": "#10B981",  # Emerald Green
        "bg_color": "#ECFDF5",
        "badge_class": "badge-low",
        "action_title": "Loyalty Maintenance",
        "description": "Customer demonstrates healthy usage patterns and strong retention indicators."
    },
    "Medium": {
        "range": (0.30, 0.70),
        "label": "Medium Risk",
        "color": "#F59E0B",  # Amber
        "bg_color": "#FFFBEB",
        "badge_class": "badge-medium",
        "action_title": "Proactive Engagement",
        "description": "Customer shows early warning signals of disengagement or friction."
    },
    "High": {
        "range": (0.70, 1.00),
        "label": "High Risk",
        "color": "#EF4444",  # Crimson Red
        "bg_color": "#FEF2F2",
        "badge_class": "badge-high",
        "action_title": "Immediate Retention Intervention",
        "description": "Customer is highly likely to churn. Urgent high-touch intervention required."
    }
}


def get_risk_tier(probability: float) -> Tuple[str, Dict[str, Any]]:
    """
    Categorizes churn probability into risk tiers.
    
    Args:
        probability: Float between 0.0 and 1.0.
        
    Returns:
        Tuple of (risk_key, risk_metadata_dict).
    """
    if probability <= 0.30:
        return "Low", RISK_CATEGORIES["Low"]
    elif probability <= 0.70:
        return "Medium", RISK_CATEGORIES["Medium"]
    else:
        return "High", RISK_CATEGORIES["High"]


def save_artifact(obj: Any, file_path: Path) -> None:
    """Save an object using joblib."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, file_path)
    logger.info(f"Artifact successfully saved to: {file_path}")


def load_artifact(file_path: Path) -> Any:
    """Load an object using joblib."""
    if not file_path.exists():
        raise FileNotFoundError(f"Requested artifact not found at: {file_path}")
    return joblib.load(file_path)


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """Save dictionary to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON data successfully saved to: {file_path}")


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found at: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_currency(value: float) -> str:
    """Format float as currency string."""
    return f"${value:,.2f}"


def format_percent(value: float) -> str:
    """Format float as percentage string."""
    return f"{value * 100:.1f}%" if value <= 1.0 else f"{value:.1f}%"
