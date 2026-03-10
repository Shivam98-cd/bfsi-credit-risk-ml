# 🏦 BFSI Credit Risk — End-to-End ML Prediction Engine

> An end-to-end Machine Learning pipeline that predicts loan default probability across a **₹89 Cr+ BFSI portfolio** using 4 models — deployed as an interactive **Streamlit dashboard** with real-time risk scoring, KS Statistic analytics, and multi-model comparison.

---

## 📸 Dashboard Preview

<img width="1809" height="817" alt="model_dashboard png" src="https://github.com/user-attachments/assets/eca2d571-e438-4811-a9cc-e854ea363bc9" />


---

## 📌 Table of Contents
- [Business Problem](#-business-problem)
- [Project Architecture](#-project-architecture)
- [Dataset Overview](#-dataset-overview)
- [Model Performance](#-model-performance)
- [Key Findings](#-key-findings)
- [Dashboard Features](#-dashboard-features)
- [Tech Stack](#-tech-stack)
- [Folder Structure](#-folder-structure)
- [How to Run](#-how-to-run)
- [Author](#-author)

---

## 💼 Business Problem

In the BFSI sector, identifying loan defaulters **before** disbursement is critical to protecting portfolio health. Traditional rule-based systems miss complex, non-linear risk patterns.

**The core question:**
> *Given a borrower's financial profile, what is the probability they will default — and which features drive that risk?*

This project answers that question using 4 ML models, industry-standard banking evaluation metrics (KS Statistic, ROC AUC), and a production-ready Streamlit interface.

---

## 🏗️ Project Architecture

```
credit_risk_customers.xlsx
         │
         ▼
  train_model.py          ← Data loading, encoding, scaling, training
         │
         ├── encoders.pkl       (per-column LabelEncoders)
         ├── scaler.pkl         (StandardScaler)
         ├── features.pkl       (column order for inference)
         ├── X_test.pkl         (cached test features)
         ├── y_test.pkl         (cached test labels)
         ├── XGBoost.pkl
         ├── Random_Forest.pkl
         ├── Decision_Tree.pkl
         └── Logistic_Regression.pkl
                  │
                  ▼
             app.py             ← Streamlit dashboard
                  │
                  ├── Tab 1: Individual Risk Prediction
                  └── Tab 2: Model Performance Analytics
                            │
                            └── evaluate.py  (KS Stat, Confusion Matrix, ROC AUC)
```

---

## 📊 Dataset Overview

| Attribute | Detail |
|---|---|
| **Records** | 1,700 loan customers |
| **Features Used** | 11 (after dropping ID, Name, Date) |
| **Train / Test Split** | 80% / 20% → 1,360 / 340 samples |
| **Class Balance** | 66.5% No Default / 33.5% Default |
| **Missing Values** | Zero |
| **Categorical Encoding** | Per-column LabelEncoder (Employment, Loan Purpose, City) |
| **Feature Scaling** | StandardScaler on all 11 features |

### Features

| Feature | Type | Role |
|---|---|---|
| `Age` | Numeric | Demographic |
| `Annual_Income` | Numeric | Financial |
| `Employment_Status` | Categorical | Risk Indicator |
| `Credit_History_Score` | Numeric | Risk Indicator |
| `Number_of_Past_Loans` | Numeric | Behavioral |
| `Debt_to_Income_Ratio` | Numeric | **#1 Predictor** |
| `Loan_Amount` | Numeric | Financial |
| `Loan_Term` | Numeric | Loan Detail |
| `Interest_Rate` | Numeric | Loan Detail |
| `Loan_Purpose` | Categorical | Loan Detail |
| `City` | Categorical | Geographic |

---

## 🤖 Model Performance

| Model | Accuracy | ROC AUC | F1 Score | Precision | Recall |
|---|---|---|---|---|---|
| **XGBoost** ⭐ | **94.71%** | **0.9836** | **0.9167** | **0.9709** | **0.8684** |
| Random Forest | 95.59% | 0.9760 | 0.9296 | 1.0000 | 0.8684 |
| Decision Tree | 95.29% | 0.9676 | 0.9259 | 0.9804 | 0.8772 |
| Logistic Regression | 91.18% | 0.9503 | 0.8707 | 0.8559 | 0.8860 |

### 🏆 Best Model: XGBoost (Highest ROC AUC)

> In credit risk, **ROC AUC is the most important metric** — it measures how well the model separates defaulters from non-defaulters across all decision thresholds. XGBoost achieves the highest AUC of **0.9836**, making it the best model for production credit decisions.

```
Confusion Matrix — XGBoost (340 test samples)
─────────────────────────────────────────────────────
                    Predicted: No Default   Predicted: Default
Actual: No Default        223                    3
Actual: Default            15                   99
─────────────────────────────────────────────────────
→ ROC AUC 0.9836: Strongest discriminatory power across all 4 models
→ 86.8% Recall: Catches 99 out of 114 actual defaulters
→ Only 3 false positives vs Random Forest's 0 — negligible trade-off
```

---

## 💡 Key Findings

### 🔴 1. DTI Ratio is the Dominant Default Predictor — 76.8% Feature Importance

Debt-to-Income Ratio alone drives **76.8% of the Random Forest prediction** and **92.9% of the Decision Tree prediction** — and is the top split node in XGBoost trees as well — confirming the DTI inflection point identified in the companion Excel and SQL projects.

| Feature | RF Importance | DT Importance |
|---|---|---|
| **Debt_to_Income_Ratio** | **76.8%** | **92.9%** |
| Credit_History_Score | 5.4% | 4.0% |
| Loan_Amount | 3.4% | 1.6% |
| Annual_Income | 3.4% | 0.5% |
| Interest_Rate | 3.0% | 0.0% |
| All others | < 3% each | ~0% |

---

### 🔴 2. XGBoost Achieves Highest ROC AUC — Best for Credit Decisions

With **ROC AUC = 0.9836**, XGBoost has the strongest ability to rank borrowers by default risk across all decision thresholds — the gold standard metric used by BFSI risk teams. Random Forest has higher raw accuracy (95.59%) but XGBoost's superior AUC makes it the preferred production model.

---

### 🟠 3. All 4 Models Agree on Risk Signals

Despite different algorithms, all 4 models converge on DTI ratio and Credit History Score as the top predictors — providing high confidence in these signals for underwriting decisions.

---

### 🟡 4. KS Statistic Validates Discriminatory Power

The KS Statistic (Kolmogorov-Smirnov) measures how well a model separates "good" and "bad" borrowers — a standard BFSI evaluation metric. A **Max KS > 40** signals a high-performing credit model. All models in this project exceed this threshold.

---

## 🖥️ Dashboard Features

### Tab 1 — Individual Risk Prediction
- Input all 11 borrower attributes via interactive widgets
- Select any of the 4 trained models for prediction
- Real-time **default probability score** with progress bar
- Clear ✅ Low Risk / ❌ High Risk verdict

### Tab 2 — Model Performance Analytics
- **KS Statistic Table** with color-gradient highlighting
- **Confusion Matrix** heatmap
- **Multi-model comparison table** with selected model highlighted in green
- Switch between all 4 models dynamically

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **ML Framework** | Scikit-learn (Random Forest, Decision Tree, Logistic Regression), XGBoost |
| **Frontend** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Model Serialization** | Joblib |
| **Data Source** | Microsoft Excel (.xlsx) |

---

## 📁 Folder Structure

```
bfsi-credit-risk-ml/
│
├── app.py                        ⭐ Streamlit dashboard
├── train_model.py                ← Training & serialization pipeline
├── evaluate.py                   ← KS Stat, ROC AUC, Confusion Matrix
├── preprocess.py                 ← Preprocessing utilities
├── requirements.txt              ← Dependencies
├── credit_risk_customers.xlsx    ← Raw dataset
│
├── model/                        ← Generated by train_model.py
│   ├── XGBoost.pkl
│   ├── Random_Forest.pkl
│   ├── Decision_Tree.pkl
│   ├── Logistic_Regression.pkl
│   ├── encoders.pkl              ← Per-column LabelEncoders
│   ├── scaler.pkl                ← StandardScaler
│   ├── features.pkl              ← Feature column order
│   ├── X_test.pkl                ← Cached test features
│   └── y_test.pkl                ← Cached test labels
│
└── docs/
    └── dashboard_preview.png     ← Dashboard screenshot
```

---

## ▶️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/Shivam98-cd/bfsi-credit-risk-ml.git
cd bfsi-credit-risk-ml
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the models**
```bash
python train_model.py
```
This generates all 9 `.pkl` files in the `model/` folder.

**4. Launch the dashboard**
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🔮 Future Scope

- **SHAP Explainability** — Local explanations for individual predictions
- **Hyperparameter Tuning** — Optuna/GridSearchCV for model optimization
- **Cross-Validation** — K-Fold CV for more robust accuracy estimates
- **Model Drift Monitoring** — Automated alerts for accuracy degradation
- **PostgreSQL Integration** — Replace Excel with a live database

---

## 👤 Author

**Shivam Yadav**
Data Scientist | BFSI Domain Specialist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/shivam-yadav98/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Shivam98-cd)

---

> ⭐ If you found this project useful, please consider starring the repository!
