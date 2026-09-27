# 🛡️ UPI-SHIELD: Adaptive, Cost-Aware & Explainable Fraud Risk Scoring

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Model-XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Explainability-SHAP](https://img.shields.io/badge/XAI-SHAP%20TreeExplainer-green.svg)](https://shap.readthedocs.io/)
[![Tests-Pytest](https://img.shields.io/badge/Tests-14%20Passed-brightgreen.svg)](https://docs.pytest.org/)

> **UPI-SHIELD** is a comprehensive, production-grade fraud risk scoring and decision intelligence framework for digital payments. Moving beyond conventional binary classification, UPI-SHIELD combines **temporal behavioral modeling**, **cost-sensitive threshold optimization**, **adaptive 0–100 continuous risk scoring**, and **local SHAP feature attribution** into a real-time decision pipeline.

---

## 📑 Table of Contents
- [Executive Overview](#-executive-overview)
- [System Architecture](#-system-architecture)
- [Core Research Pillars](#-core-research-pillars)
- [Empirical Benchmark Results](#-empirical-benchmark-results)
- [Project Directory Structure](#-project-directory-structure)
- [Quick Start Guide](#-quick-start-guide)
- [API Reference & Payload Schema](#-api-reference--payload-schema)
- [Streamlit Intelligence Dashboard](#-streamlit-intelligence-dashboard)
- [Dataset Positioning & Governance](#-dataset-positioning--governance)

---

## 🎯 Executive Overview

Modern digital payment ecosystems operate at ultra-high transaction velocity where standard binary fraud classifiers (predicting $0$ or $1$ at a fixed $0.50$ threshold) fail on two fronts:
1. **Symmetric Loss Assumption:** They treat False Positives (user friction) and False Negatives (financial fraud loss) identically.
2. **Black-Box Opacity:** They offer no auditability or actionable diagnostic insight to payment operations teams and consumers.

**UPI-SHIELD** solves this with an end-to-end framework:
- **Asymmetric Cost Optimization:** Minimizes expected financial and operational loss ($C_{FP}=1.0, C_{FN}=10.0$), selecting an optimal decision threshold ($\tau^* = 0.09$) that delivers a **24.83% cost reduction** over standard baselines.
- **Continuous 0–100 Risk Scoring:** Maps calibrated fraud probabilities into intuitive severity tiers (**LOW**, **MODERATE**, **HIGH**, **CRITICAL**).
- **Explainability by Design:** Computes additive feature contributions via **SHAP TreeExplainer** with human-readable diagnostic messages for every scored transaction.

```
┌──────────────────┐     ┌────────────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│  Transaction     │ ──> │ Feature Engineering    │ ──> │ XGBoost Inference    │ ──> │ Probability &          │
│  Request Payload │     │ & Preprocessing        │     │ Classifier           │     │ Cost-Aware Policy (τ*) │
└──────────────────┘     └────────────────────────┘     └──────────────────────┘     └────────────────────────┘
                                                                                                  │
                                 ┌────────────────────────────────────────────────────────────────┘
                                 ▼
                     ┌──────────────────────┐     ┌────────────────────────┐
                     │ 0–100 Risk Score     │ ──> │ SHAP Local Feature     │ ──> [ FastAPI & Streamlit UI ]
                     │ & 4 Severity Bands   │     │ Attribution Report     │
                     └──────────────────────┘     └────────────────────────┘
```

---

## 🔬 Core Research Pillars

### 1. Chronological Splitting & Leakage Prevention
Transactions are strictly split along chronological time boundaries (`train.csv` $\rightarrow$ `validation.csv` $\rightarrow$ `test.csv`) to mirror real-world out-of-time deployment and prevent future data leakage.

### 2. Behavioral & Velocity Feature Engineering
- **Time-Cycle Decomposition:** Extracted transaction hour, minute, day, and weekday cycles.
- **Transactional Signatures:** Log-transformed amount, decimal component extraction, email domain relationship matching ($P\_email$ vs $R\_email$).
- **Missingness Profiling:** Missing address and card attribute counters serving as fraud indicators.

### 3. Cost-Aware Threshold Optimization
Fraud detection is cost-asymmetric. UPI-SHIELD sweeps threshold candidates $\tau \in [0.01, 0.99]$ on validation data to minimize total expected risk cost:
$$\text{Cost}(\tau) = C_{FP} \cdot \text{FP}(\tau) + C_{FN} \cdot \text{FN}(\tau)$$
Evaluating at $C_{FP} = 1.0$ and $C_{FN} = 10.0$ yields an optimal threshold **$\tau^* = 0.09$**, reducing total cost from $28,504$ units down to $21,427$ units (**$24.83\%$ financial savings**).

### 4. Adaptive 0–100 Risk Scoring Matrix
Calibrated fraud probabilities $P(Y=1)$ are mapped to a normalized 0–100 risk scale categorized into actionable operational tiers:

| Risk Tier | Score Range | Operational Action |
|:---|:---:|:---|
| 🟢 **LOW** | 0 – 24 | Frictionless instant clearance (Auto-Approve) |
| 🟡 **MODERATE** | 25 – 49 | Standard real-time transaction monitoring |
| 🟠 **HIGH** | 50 – 74 | Step-up authentication (2FA / Biometric verification) |
| 🔴 **CRITICAL** | 75 – 100 | Immediate transaction block & compliance review |

### 5. Local SHAP TreeExplainer Interpretability
Every transaction is decomposed into exact additive log-odds contributions, exposing top risk-escalating factors ($\uparrow$) and trust-building factors ($\downarrow$) alongside plain-language diagnostic descriptions.

---

## 📊 Empirical Benchmark Results

All figures are reproduced directly from stored chronological experiment artifacts (`experiments/results/`):

### A. Model Performance Comparison
| Metric | Logistic Regression (Baseline) | XGBoost (UPI-SHIELD Engine) | Generalization Status |
|:---|:---:|:---:|:---:|
| **Validation PR-AUC** | `0.0914` | **`0.2745`** | $+200.3\%$ lift |
| **Test PR-AUC (OOT)** | `0.0914` | **`0.2081`** | $+127.7\%$ lift |
| **Validation ROC-AUC** | `0.6806` | **`0.8247`** | $+21.2\%$ lift |
| **Test ROC-AUC (OOT)** | `0.6854` | **`0.8084`** | $+17.9\%$ lift |
| **Validation Brier Score** | `0.2864` | **`0.0286`** | $10\times$ lower calibration loss |
| **Test Brier Score (OOT)** | `0.3212` | **`0.0305`** | $10\times$ lower calibration loss |

> *Note: Out-of-Time (OOT) evaluation confirms XGBoost superior discrimination and calibration stability across future temporal distributions.*

### B. Cost Sensitivity Analysis
| Cost Ratio ($C_{FP} : C_{FN}$) | Optimal Threshold ($\tau^*$) | Validation Total Cost | Precision | Recall |
|:---:|:---:|:---:|:---:|:---:|
| **1 : 5** | `0.15` | `12,192` units | `31.48%` | `35.14%` |
| **1 : 10 (Deployed)** | **`0.09`** | **`21,427` units** | **`20.97%`** | **`47.44%`** |
| **1 : 20** | `0.05` | `35,444` units | `13.40%` | `61.67%` |
| **1 : 50** | `0.02` | `57,934` units | `06.66%` | `86.00%` |

---

## 📁 Project Directory Structure

```text
upi-fraud-risk-scoring/
├── app/
│   ├── main.py                  # FastAPI server application
│   ├── predictor.py             # Inference engine & SHAP pipeline
│   ├── schemas.py               # Pydantic request/response schemas
│   ├── dashboard.py             # Streamlit operations intelligence portal
│   ├── dashboard_components.py  # Reusable UI component library & CSS design tokens
│   ├── dashboard_charts.py      # Plotly fintech visualization builders
│   ├── dashboard_data.py        # Experiment artifacts data loader
│   └── dashboard_api.py         # HTTP client for FastAPI backend
│
├── data/
│   ├── raw/                     # Raw benchmark transaction datasets
│   └── processed/               # Chronological train/validation/test splits
│
├── experiments/
│   └── results/                 # Verified experiment JSON evaluation outputs
│       ├── baseline_results.json
│       ├── xgboost_results.json
│       ├── cost_sensitive_threshold_results.json
│       ├── cost_sensitivity_results.json
│       ├── calibration_results.json
│       ├── calibration_test_results.json
│       └── explanation_report.json
│
├── models/                      # Serialized ML artifacts & policy configs
│   ├── preprocessor.joblib      # Fitted scikit-learn preprocessing pipeline
│   ├── xgboost_fraud_model.joblib # Trained XGBoost model
│   ├── decision_policy.json     # Cost-aware decision threshold parameters
│   └── risk_scoring_config.json # 0–100 risk band definitions
│
├── src/
│   ├── data/                    # Dataset download and chronological splitting
│   ├── features/                # Behavioral & velocity feature engineering
│   ├── models/                  # Baseline and XGBoost training scripts
│   ├── calibration/             # Isotonic probability calibration scripts
│   ├── decision/                # Cost-sensitive threshold optimizer
│   ├── risk_scoring/            # Probability to 0-100 score mapping
│   └── explainability/          # SHAP TreeExplainer & explanation formatters
│
├── tests/                       # Automated pytest test suites
│   ├── test_data_split.py       # Zero-leakage temporal order tests
│   ├── test_pipeline.py         # End-to-end model inference & banding tests
│   └── test_reproducibility.py  # Artifact integrity verification tests
│
├── requirements.txt             # Python dependencies
├── AGENTS.md                    # Research rules & dataset positioning
└── README.md                    # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Clone & Set Up Environment
```powershell
# Clone the repository
git clone https://github.com/prajktanandurkar09/upi-fraud-risk-scoring.git
cd upi-fraud-risk-scoring

# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\activate
# Or macOS/Linux:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
```powershell
.venv\Scripts\python -m pytest
```

### 3. Launch the Backend API Service
```powershell
.venv\Scripts\python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Interactive Swagger docs will be live at: `http://127.0.0.1:8000/docs`

### 4. Launch the Streamlit Intelligence Dashboard
In a separate terminal window:
```powershell
.venv\Scripts\python -m streamlit run app/dashboard.py
```
Open `http://localhost:8501` in your browser.

---

## 🔌 API Reference & Payload Schema

### `POST /predict`
Evaluates a transaction payload and returns risk score, band, cost-aware decision, and top SHAP explanations.

#### Sample Request (`application/json`):
```json
{
  "TransactionDT": 86400.0,
  "TransactionAmt": 25.0,
  "ProductCD": "W",
  "card1": 10000.0,
  "card2": 111.0,
  "card3": 150.0,
  "card4": "visa",
  "card5": 226.0,
  "card6": "debit",
  "addr1": 100.0,
  "addr2": 87.0,
  "P_emaildomain": "gmail.com",
  "R_emaildomain": "gmail.com",
  "DeviceType": "mobile",
  "DeviceInfo": "Android"
}
```

#### Sample Response (`application/json`):
```json
{
  "fraud_probability": 0.0084,
  "risk_score": 1,
  "risk_band": "LOW",
  "decision_threshold": 0.09,
  "decision": "LEGITIMATE",
  "probability_source": "raw_xgboost",
  "explanations": [
    {
      "feature": "Transaction amount",
      "technical_feature": "numeric__TransactionAmt",
      "shap_value": -0.635175,
      "direction": "decreases_risk",
      "symbol": "↓",
      "message": "↓ Transaction amount decreased the model's risk prediction"
    },
    {
      "feature": "Email domain relationship",
      "technical_feature": "numeric__email_domain_match",
      "shap_value": 0.430792,
      "direction": "increases_risk",
      "symbol": "↑",
      "message": "↑ Email domain relationship increased the model's risk prediction"
    }
  ]
}
```

---

## 💻 Streamlit Intelligence Dashboard

The frontend operations cockpit provides 4 primary operational views:

1. **⚡ Risk Analyzer:**
   - 1-Click test scenario presets (Low-Risk P2P, Moderate E-Commerce, Velocity Spike, Critical Cross-Border).
   - Real-time Plotly neon radial risk gauge (0–100 score).
   - Executive decision banners (🚨 `FRAUD` vs 🛡️ `LEGITIMATE`).
   - Compact local SHAP diagnostic meters.
2. **📈 Risk Analytics:**
   - Out-of-time benchmark performance charts (PR-AUC, ROC-AUC, Brier score).
   - Cost vs. threshold optimization curves with diamond optimal marker callout ($\tau^* = 0.09$).
   - Cost sensitivity ratio breakdowns and model comparison matrices.
3. **🔎 Explainability:**
   - SHAP TreeExplainer diverging waterfall horizontal bar chart.
   - Positive fraud pushers vs negative legitimacy anchors impact summary.
   - Tabular feature evidence and technical token mappings.
4. **⚙️ Model & System:**
   - End-to-end 9-stage prediction pipeline flowchart.
   - Risk band classification criteria and framework governance policies.

---

## 📜 Dataset Positioning & Governance

> **Important Research Notice:**  
> The **IEEE-CIS Fraud Detection Dataset** is used as an open, reproducible benchmark dataset for developing and evaluating this UPI-style digital payment fraud-risk scoring framework. The model is evaluated on benchmark transaction distributions and is not trained on proprietary Indian UPI production transaction logs.

- **Reproducibility:** All experimental metrics, curves, and figures are 100% code-generated and persisted in `experiments/results/`.
- **Zero Future Leakage:** Evaluated exclusively on out-of-time chronological test partitions.
- **Fair Metric Representation:** PR-AUC (Average Precision), ROC-AUC, Brier score, and Expected Cost are prioritized over raw accuracy.

---

## 📄 License & Attribution

Developed under academic research standards for intelligent risk management in digital payment systems. Distributed under the MIT License.
