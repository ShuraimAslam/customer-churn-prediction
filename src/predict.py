"""
Prediction & Prescriptive Analytics Engine for Customer Churn Intelligence Dashboard.
Handles single-customer and batch inference, churn probability calculation,
risk tier categorization, personalized churn driver attribution, and retention playbooks.
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

from src.utils import (
    logger,
    CHURN_MODEL_PATH,
    SCALER_PATH,
    FEATURES_PATH,
    MODELS_DIR,
    get_risk_tier,
    load_artifact,
    load_json
)
from src.feature_engineering import add_engineered_features, encode_features, align_features


class ChurnPredictor:
    """
    Production-ready Customer Churn Inference Engine.
    Loads persisted artifacts (model, scaler, feature list) and executes
    risk classification, explainability, and retention recommendation generation.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.available_models = {}
        self.is_loaded = False
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads model, scaler, and feature names from disk."""
        try:
            self.model = load_artifact(CHURN_MODEL_PATH)
            self.scaler = load_artifact(SCALER_PATH)
            self.feature_columns = load_json(FEATURES_PATH)["features"]
            
            # Index available models
            self.available_models = {
                "Logistic Regression (Class Weighted)": self.model
            }
            rf_path = MODELS_DIR / "random_forest_model.pkl"
            if rf_path.exists():
                try:
                    self.available_models["Random Forest (Class Weighted)"] = load_artifact(rf_path)
                except Exception as rf_err:
                    logger.warning(f"Could not load Random Forest model: {rf_err}")

            self.is_loaded = True
            logger.info(f"ChurnPredictor initialized with models: {list(self.available_models.keys())}")
        except Exception as e:
            logger.warning(f"Predictor artifacts not ready yet: {str(e)}")
            self.is_loaded = False

    def predict_customer(self, customer_data: Dict[str, Any], model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculates churn probability, risk tier, top drivers, and retention actions for a single customer.
        
        Args:
            customer_data: Dictionary containing customer attributes
            model_name: Optional name of the model to use from available_models
            
        Returns:
            Dictionary containing prediction metrics and business insights.
        """
        if not self.is_loaded:
            self._load_artifacts()
            if not self.is_loaded:
                raise RuntimeError("Predictor cannot run: Model artifacts are missing. Run train.py first.")

        # Convert to DataFrame
        df_single = pd.DataFrame([customer_data])

        # Fill missing standard defaults if not passed
        default_fields = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.0,
            "TotalCharges": 70.0
        }
        for k, v in default_fields.items():
            if k not in df_single.columns:
                df_single[k] = v

        # Feature engineering
        df_feat = add_engineered_features(df_single)

        # One-hot encode with alignment
        df_encoded, _ = encode_features(df_feat, is_training=False, expected_columns=self.feature_columns)

        # Numerical scaling
        numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AverageMonthlySpend", "ServiceCount"]
        cols_to_scale = [c for c in numerical_cols if c in df_encoded.columns]
        df_scaled = df_encoded.copy()
        df_scaled[cols_to_scale] = self.scaler.transform(df_encoded[cols_to_scale])

        # Choose model
        active_model = self.available_models.get(model_name, self.model) if model_name else self.model

        # Inference
        probability = float(active_model.predict_proba(df_scaled)[0, 1])
        prediction = int(probability >= 0.50)

        # Determine risk tier
        risk_key, risk_meta = get_risk_tier(probability)

        # Identify personalized churn drivers (Rule-based & feature contribution)
        drivers = self._extract_churn_drivers(customer_data, probability)

        # Generate targeted retention recommendations
        recommendations = self._generate_retention_recommendations(customer_data, risk_key, drivers)

        return {
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(probability, 4),
            "churn_probability_pct": round(probability * 100, 1),
            "risk_tier": risk_key,
            "risk_label": risk_meta["label"],
            "risk_color": risk_meta["color"],
            "risk_bg_color": risk_meta["bg_color"],
            "description": risk_meta["description"],
            "model_used": model_name or "Logistic Regression (Class Weighted)",
            "drivers": drivers,
            "recommendations": recommendations,
            "customer_profile": customer_data
        }

    def predict_batch(self, df_batch: pd.DataFrame, model_name: Optional[str] = None) -> pd.DataFrame:
        """
        Runs batch prediction on a customer dataset.
        Returns the original DataFrame enriched with:
        - Churn_Probability_Pct (%)
        - Risk_Tier (Low Risk, Medium Risk, High Risk)
        - Predicted_Churn (Yes / No)
        """
        if not self.is_loaded:
            self._load_artifacts()
            if not self.is_loaded:
                raise RuntimeError("Predictor cannot run: Model artifacts are missing.")

        df_feat = add_engineered_features(df_batch)
        df_encoded, _ = encode_features(df_feat, is_training=False, expected_columns=self.feature_columns)
        
        numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AverageMonthlySpend", "ServiceCount"]
        cols_to_scale = [c for c in numerical_cols if c in df_encoded.columns]
        df_scaled = df_encoded.copy()
        df_scaled[cols_to_scale] = self.scaler.transform(df_encoded[cols_to_scale])

        active_model = self.available_models.get(model_name, self.model) if model_name else self.model
        probs = active_model.predict_proba(df_scaled)[:, 1]

        results_df = df_batch.copy()
        results_df["Churn_Probability_Pct"] = (probs * 100).round(1)
        results_df["Predicted_Churn"] = np.where(probs >= 0.50, "Yes", "No")

        def assign_tier(p):
            if p <= 0.30: return "Low Risk"
            elif p <= 0.70: return "Medium Risk"
            else: return "High Risk"

        results_df["Risk_Tier"] = [assign_tier(p) for p in probs]
        return results_df

    def _extract_churn_drivers(self, data: Dict[str, Any], probability: float) -> List[Dict[str, str]]:
        """
        Diagnoses specific human-readable churn drivers based on the customer's attributes.
        """
        drivers = []
        tenure = float(data.get("tenure", 1))
        monthly = float(data.get("MonthlyCharges", 50.0))
        contract = data.get("Contract", "Month-to-month")
        internet = data.get("InternetService", "DSL")
        payment = data.get("PaymentMethod", "Electronic check")
        tech_support = data.get("TechSupport", "No")
        online_security = data.get("OnlineSecurity", "No")
        paperless = data.get("PaperlessBilling", "Yes")

        # Driver 1: Month-to-month contract
        if contract == "Month-to-month":
            drivers.append({
                "factor": "Short-Term Commitment",
                "severity": "High",
                "detail": "Customer is on a Month-to-Month contract with zero contractual lock-in or cancellation penalties."
            })

        # Driver 2: Tenure window
        if tenure <= 6:
            drivers.append({
                "factor": "Early Flight Stage",
                "severity": "High",
                "detail": f"Customer has been active for only {int(tenure)} month(s). New subscribers in the first 6 months have the highest attrition rate."
            })
        elif tenure <= 15:
            drivers.append({
                "factor": "Formative Tenure Window",
                "severity": "Medium",
                "detail": f"Customer tenure ({int(tenure)} months) is still below the established loyalty threshold of 24 months."
            })

        # Driver 3: High Monthly Bill
        if monthly >= 80.0:
            drivers.append({
                "factor": "High Monthly Spend Friction",
                "severity": "High",
                "detail": f"Monthly billing of ${monthly:.2f} is in the top quartile of subscriber fees, creating high sensitivity to price."
            })
        elif monthly >= 65.0:
            drivers.append({
                "factor": "Elevated Monthly Cost",
                "severity": "Medium",
                "detail": f"Monthly charges of ${monthly:.2f} create ongoing sensitivity to competitor promotion offers."
            })

        # Driver 4: Fiber Optic Disconnect
        if internet == "Fiber optic" and tech_support != "Yes":
            drivers.append({
                "factor": "Fiber Optic without Technical Support",
                "severity": "High",
                "detail": "Subscribes to premium Fiber Optic internet without dedicated Tech Support add-on; technical hiccups trigger rapid defection."
            })

        # Driver 5: Payment method friction
        if payment == "Electronic check":
            drivers.append({
                "factor": "Electronic Check Vulnerability",
                "severity": "Medium",
                "detail": "Electronic check payments experience highest manual payment friction and involuntary churn compared to automated cards."
            })

        # Driver 6: Lack of online security
        if online_security == "No" and internet != "No":
            drivers.append({
                "factor": "No Security Add-on",
                "severity": "Low",
                "detail": "Subscriber has not activated Online Security, diminishing perceived ecosystem value and loyalty lock-in."
            })

        # If probability is low and few risk drivers exist, highlight positive anchors
        if not drivers or probability <= 0.30:
            drivers.append({
                "factor": "Healthy Account Loyalty Anchor",
                "severity": "Positive",
                "detail": f"Substantial tenure ({int(tenure)} months) combined with {contract} agreement creates robust retention stability."
            })

        return drivers[:4]

    def _generate_retention_recommendations(
        self,
        data: Dict[str, Any],
        risk_tier: str,
        drivers: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Produces targeted, high-impact business retention playbooks based on risk tier and drivers.
        """
        contract = data.get("Contract", "Month-to-month")
        tenure = float(data.get("tenure", 1))
        monthly = float(data.get("MonthlyCharges", 50.0))
        internet = data.get("InternetService", "DSL")
        payment = data.get("PaymentMethod", "Electronic check")
        
        recommendations = []

        if risk_tier == "High":
            recommendations.append({
                "action": "Contract Upgrade Incentive",
                "badge": "Immediate Priority",
                "description": "Offer an attractive 15% discount on an annual contract commitment to transition away from Month-to-month vulnerability.",
                "expected_impact": "Reduces immediate churn hazard by ~45%."
            })
            if internet == "Fiber optic" or monthly >= 70.0:
                recommendations.append({
                    "action": "Complimentary Value-Add Bundle",
                    "badge": "Service Lock-in",
                    "description": "Provide free 6-month Tech Support and Online Security bundle to elevate perceived value and service satisfaction.",
                    "expected_impact": "Increases customer satisfaction score by ~30%."
                })
            recommendations.append({
                "action": "Dedicated Relationship Manager Call",
                "badge": "High-Touch Outreach",
                "description": "Schedule a proactive customer success call within 48 hours to resolve any billing or connection grievances.",
                "expected_impact": "Directly resolves underlying dissatisfaction for 60% of at-risk subscribers."
            })

        elif risk_tier == "Medium":
            recommendations.append({
                "action": "Targeted Loyalty Promotion",
                "badge": "Engagement",
                "description": "Issue a personalized promotional credit ($10/month for 3 months) conditioned on maintaining active status.",
                "expected_impact": "Extends customer lifespan past initial churn friction window."
            })
            if payment == "Electronic check":
                recommendations.append({
                    "action": "Automated Billing Incentive",
                    "badge": "Payment Friction",
                    "description": "Offer a one-time $25 billing credit for switching to automated credit card or bank transfer billing.",
                    "expected_impact": "Lowers payment dropouts by ~28%."
                })
            recommendations.append({
                "action": "Mid-Tenure Experience Check",
                "badge": "Proactive Care",
                "description": "Send a digital 2-question satisfaction survey offering immediate live chat assistance if negative sentiment is indicated.",
                "expected_impact": "Identifies early friction points before escalation."
            })

        else:  # Low Risk
            recommendations.append({
                "action": "Maintain Current Service Cadence",
                "badge": "Loyalty",
                "description": "Maintain standard relationship workflow without aggressive discounts; customer is satisfied with current setup.",
                "expected_impact": "Avoids unnecessary discounting margin erosion."
            })
            recommendations.append({
                "action": "Referral & Advocacy Program",
                "badge": "Growth",
                "description": "Invite customer into the VIP Referral Program: reward both customer and referred peers with billing credits.",
                "expected_impact": "Leverages satisfied accounts to acquire new customers at low CAC."
            })
            recommendations.append({
                "action": "Annual Loyalty Milestone Recognition",
                "badge": "Appreciation",
                "description": "Deliver a tenure anniversary recognition perk (e.g. speed boost or streaming credit) to reinforce brand affinity.",
                "expected_impact": "Solidifies long-term NPS and brand advocacy."
            })

        return recommendations


def get_default_customer_input() -> Dict[str, Any]:
    """Returns a sensible default customer record for UI pre-population."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.85,
        "TotalCharges": 239.55
    }
