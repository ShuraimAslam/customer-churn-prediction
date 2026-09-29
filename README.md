# Customer Churn Intelligence Dashboard

A full-stack enterprise machine learning application for predicting, analyzing, and mitigating customer churn in the telecommunications sector — built as a portfolio-quality data science project.

---

## 🚀 Live Application

Runs locally via **Streamlit** at `http://localhost:8501`

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 📈 Executive Overview | KPI cards, churn distribution, contract analysis, strategic takeaways |
| 🔍 Churn Analytics | Interactive EDA with cohort filters, tenure/pricing dynamics, heatmaps |
| 🔮 Customer Churn Prediction | Single-profile prediction + batch CSV scoring w/ multi-model support |
| 🧬 Churn Drivers | ML feature importance with business interpretations |
| 🛡️ Retention Center | Prescriptive playbooks + ROI calculator |
| ⚙️ Model Performance | Algorithm leaderboard, confusion matrices, ROC curves |

---

## 🧠 ML Models

- **Logistic Regression (Class Weighted)** — Best model (ROC-AUC: 0.846, Recall: 78.6%)
- **Random Forest (Class Weighted)** — Benchmarked for comparison
- Both trained with SMOTE vs class-weighting imbalance strategy comparison

---

## 🗂️ Project Structure

```
customer-churn-prediction/
├── app/
│   └── streamlit_app.py        # Main Streamlit dashboard (6 pages)
├── assets/
│   └── style.css               # Custom UI design system
├── data/
│   ├── raw/                    # Original Telco Churn CSV
│   └── processed/              # Cleaned + engineered train/test splits
├── models/
│   ├── churn_model.pkl         # Production model (Logistic Regression)
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.json
│   ├── feature_importance.json
│   └── model_metrics.json
├── notebooks/                  # (Reserved for EDA notebooks)
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py  # AverageMonthlySpend, ServiceCount, etc.
│   ├── train.py                # Full training pipeline
│   ├── evaluate.py
│   ├── predict.py              # ChurnPredictor (single + batch inference)
│   └── utils.py
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Setup & Run

```bash
# 1. Clone repository
git clone https://github.com/ShuraimAslam/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Retrain models
python src/train.py

# 5. Launch the dashboard
streamlit run app/streamlit_app.py
```

---

## 📦 Key Dependencies

- `streamlit` — Dashboard framework
- `scikit-learn` — ML models & preprocessing
- `imbalanced-learn` — SMOTE oversampling
- `plotly` — Interactive visualizations
- `pandas`, `numpy` — Data processing
- `joblib` — Model serialization

---

## 📈 Dataset

**IBM Telco Customer Churn** dataset (7,043 customers, 21 features).

Key features: `tenure`, `Contract`, `InternetService`, `MonthlyCharges`, `PaymentMethod`, `TechSupport`, `OnlineSecurity`, and more.

---

## 🔑 Critical Bug Fix Applied

A one-hot encoding bug in `src/feature_engineering.py` was identified and fixed:
> When running inference on a **single-row DataFrame**, `pd.get_dummies(..., drop_first=True)` would drop the only category value, producing all-zero feature vectors — causing the model to underestimate churn probability significantly (e.g. High Risk profiles scoring as Medium/Low Risk).

**Fix:** Use `drop_first=False` during inference only, then align to the training feature schema via `align_features()`.

---

## 👤 Author

**ShuraimAslam** — Built with Python, Scikit-Learn, Plotly & Streamlit.
