# 🏡 Real Estate Investment Advisor — Predicting Property Profitability & Future Value

[![Python Version](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-blue?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-brightgreen)](https://lightgbm.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-3.17.2-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66.1-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![DagsHub](https://img.shields.io/badge/MLOps-DagsHub%20Cloud-198754)](https://dagshub.com/)

An enterprise-grade, end-to-end Machine Learning intelligence platform designed to evaluate Indian residential real estate listings, predict investment profitability (**Good Investment vs. Cautious Avoid**), forecast **5-year property valuations & annualized CAGR trajectories**, and stream real-time operational telemetry to a remote **DagsHub MLflow** cloud tracking server.

---

## 📌 Executive Highlights & System Benchmarks

| Domain Metric | Project Outcome | Details |
| :--- | :--- | :--- |
| **Dataset Scale** | **250,000 Listings** | Pan-Indian coverage across 42 cities, 20 states, 0 missing values. |
| **Classification Champion** | **XGBoost (v1)** | **99.24% Accuracy**, **0.9924 F1-Score**, **0.9996 ROC-AUC** on 50,000 test listings. |
| **Regression Champion** | **LightGBM (v1)** | **$R^2 = 0.9999$**, **RMSE = ₹1.76 Lakhs**, **MAE = ₹1.34 Lakhs**, **MAPE = 0.63%**. |
| **Serving Latency** | **p50 = 37.56 ms** | In-memory cached pipelines delivering sub-40ms end-to-end predictions. |
| **MLOps Governance** | **Remote Cloud MLflow** | Registered in Model Registry under stage `"Production"` with live audit streaming. |
| **Decision Portal** | **Streamlit Single-Page** | Reactive location auto-sync, 1-click presets, Plotly 5-year growth curve, civic scorecard. |

---

## 🎯 Problem Statement & Core Value Proposition

Indian residential real estate is a $250B+ market marked by significant information asymmetry, localized pricing volatility, and marketing markups. Stakeholders face three primary risks:
1. **Pricing Inflation:** Buying at rates substantially above neighborhood micro-market medians.
2. **Liquidity Traps:** Acquiring non-standard configurations with sluggish resale liquidity.
3. **Execution & Delivery Risk:** Under-construction properties with possession delays and deficient infrastructure.

**The Real Estate Investment Advisor solves this by delivering dual-engine intelligence:**
- **Task A (Classification):** Evaluates a property listing on an objective 100-point composite investment scorecard (Price Attractiveness, Civic Infrastructure Density, BHK Configuration Liquidity, and Possession Status) to produce a calibrated investment verdict with model confidence.
- **Task B (Regression):** Predicts long-term capital appreciation over a 5-year investment horizon (`Future_Price_5Y` in ₹ Lakhs) and models the annualized compound annual growth rate (CAGR).

---

## 🔬 Exploratory Data Analysis (EDA) — Key Insights

The system incorporates findings from a 20-question exploratory investigation documented in [`notebooks/02_eda_analysis.ipynb`](notebooks/02_eda_analysis.ipynb) and visual artifacts in `reports/eda/`:

- **Macro Pricing (Q1–Q5):** Positive skew with a median listing price of ₹125.0 Lakhs (mean: ₹142.3 Lakhs). Villas command the highest median price per sq.ft (₹11,420/sqft), followed by Independent Houses (₹9,850/sqft) and Apartments (₹8,920/sqft).
- **Geographic Hierarchy (Q6–Q10):** Maharashtra (Mumbai: ₹210L), Karnataka (Bangalore: ₹145L), and Delhi NCR (Gurgaon: ₹138L) lead baseline price benchmarks. Top premium micro-markets include *Worli* (Mumbai), *Indiranagar* (Bangalore), and *Golf Course Road* (Gurgaon).
- **Feature Correlations (Q11):** `Good_Investment` correlates strongly negatively with `Price_in_Lakhs` ($r = -0.653$) and `Price_per_SqFt` ($r = -0.628$), penalizing overpriced listings, and correlates positively with `Size_in_SqFt` ($r = +0.490$).
- **Civic Infrastructure & Amenities (Q16–Q20):** High transit accessibility commands a **22.4% price premium** over low-accessibility listings and yields a higher proportion of Good Investments (50.31% vs 49.38%). Dedicated parking adds a **14.8% price premium**, and each major amenity (Gym, Pool, Security) contributes an average of **₹320–₹580/sq.ft** to baseline value.

---

## 🏆 Production Machine Learning Leaderboards

Models were trained on 200,000 listings and benchmarked on 50,000 held-out test listings using modular Scikit-Learn pipelines that completely prevent data leakage:

### 1. Classification Models Leaderboard (`Good_Investment`)

| Rank | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **XGBoost Classifier** | **99.24%** | **99.31%** | **99.17%** | **0.9924** | **0.9996** | **Champion (Production)** |
| 🥈 | **LightGBM Classifier** | 99.18% | 99.25% | 99.11% | 0.9918 | 0.9995 | Candidate |
| 🥉 | **Random Forest Classifier** | 98.91% | 99.04% | 98.78% | 0.9891 | 0.9992 | Candidate |
| 4 | **Logistic Regression** | 94.62% | 94.81% | 94.41% | 0.9461 | 0.9882 | Linear Baseline |

### 2. Regression Models Leaderboard (`Future_Price_5Y` in ₹ Lakhs)

| Rank | Model | RMSE (₹ Lakhs) | MAE (₹ Lakhs) | $R^2$ Score | MAPE (%) | Inference SLA | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **LightGBM Regressor** | **₹1.76 L** | **₹1.34 L** | **0.9999** | **0.63%** | **22 ms** | **Champion (Production)** |
| 🥈 | **XGBoost Regressor** | ₹1.89 L | ₹1.41 L | 0.9998 | 0.68% | 28 ms | Candidate |
| 🥉 | **Random Forest Regressor** | ₹2.84 L | ₹1.92 L | 0.9996 | 0.89% | 42 ms | Candidate |
| 4 | **Ridge Regression** | ₹8.42 L | ₹6.15 L | 0.9972 | 2.18% | 18 ms | Linear Baseline |

---

## ☁️ Cloud MLOps & MLflow Architecture

- **Remote Tracking Server:** Connected to DagsHub MLflow tracking server via HTTPS.
- **Model Registry Staging:** Both champions are registered in the MLflow Model Registry:
  - `Property_Classifier_Prod` (v1 | XGBoost) $\rightarrow$ Stage: `"Production"` (Alias: `champion`)
  - `Property_Regressor_Prod` (v1 | LightGBM) $\rightarrow$ Stage: `"Production"` (Alias: `champion`)
- **Real-Time Live Inference Telemetry:** Every evaluation performed through the web app streams:
  - **Parameters:** City, State, Locality, BHK, Size_in_SqFt, Listing Price, Year Built, Amenities.
  - **Metrics:** Verdict (`1`/`0`), Confidence %, Predicted 5Y Price, Projected CAGR %, and Inference Latency (ms).
  - **Tags:** Model version, timestamp, environment, audit identifier.
  - **Experiment:** Logged under `Live_Inference_Monitoring` (Experiment ID `2`).

---

## 💻 Streamlit Web Application Portal

The platform interface in [`app/app.py`](app/app.py) provides a streamlined user experience:
1. **Dynamic Location Auto-Sync:** Selecting a City (e.g. *Bhubaneswar*) instantly and reactively syncs the State (*Odisha*) and auto-suggests well-known local neighborhoods (*Patia*).
2. **1-Click Demo Presets:** Test prime metropolitan assets (*Bangalore 3BHK*), high-growth suburban assets (*Pune 2BHK*), or speculative overpriced assets (*Mumbai 1BHK*).
3. **Executive Verdict & Metrics:** Highlight banner displaying the verdict and confidence, followed by 4 cards (Current Valuation, 5-Year Projection, CAGR, and Model Confidence).
4. **Interactive 5-Year Valuation Curve:** Dynamic Plotly visualization of projected asset growth over time.
5. **Civic Infrastructure Scorecard:** 100-point rating aggregating schools, healthcare, public transit, parking, and security.
6. **Live Audit Link:** Clickable button opening the exact live inference run on the remote DagsHub MLflow tracking server.

---

## 📂 Project Repository Structure

```
Real Estate Investment Advisor/
├── app/
│   └── app.py                     # Streamlit Interactive Web Application
├── src/
│   ├── preprocessing.py           # Scikit-Learn Modular Feature Engineering Pipeline
│   ├── train.py                   # Multi-Model Training, Cross-Validation & MLflow Logging
│   ├── mlflow_service.py          # Dynamic Model Serving, In-Memory Caching & Cloud Telemetry
│   └── utils.py                   # Global Pathing, Constants, Directory Initialization
├── notebooks/
│   ├── 01_data_preprocessing.ipynb # Phase 1: Data Cleaning, Outliers & Target Formulation
│   └── 02_eda_analysis.ipynb      # Phase 2: Comprehensive 20-Question Exploratory Analysis
├── tests/
│   └── test_e2e_pipeline.py       # Phase 6: Automated End-to-End Testing Suite
├── scripts/
│   └── register_on_dagshub.py     # DagsHub MLflow Cloud Model Registration Utility
├── data/
│   ├── raw/                       # Original housing dataset
│   └── processed/
│       └── cleaned_housing_prices.csv # Cleaned 250,000-listing master dataset
├── models/
│   ├── classification/            # Champion XGBoost pipeline weights (.joblib)
│   └── regression/                # Champion LightGBM pipeline weights (.joblib)
├── reports/
│   ├── eda/                       # 20 High-Resolution Exploratory Visual Artifacts (.png)
│   ├── classification/            # Confusion matrices, ROC curves & comparison charts
│   ├── regression/                # Residual plots, actual vs predicted & comparison charts
│   └── Real_Estate_Investment_Advisor_Comprehensive_Project_Report.docx
├── .env.example                   # Template for DagsHub MLflow cloud credentials
├── .gitignore                     # Production Git exclusion rules
├── requirements.txt               # Pinned Python package dependencies
└── README.md                      # Comprehensive Project Documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python **3.11+** installed.
- Git installed.

### 2. Clone Repository & Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/nathdhiman005-svg/Real-Estate-Investment-Advisor-Predicting-Property-Profitability-Future-Value.git
cd "Real Estate Investment Advisor Predicting Property Profitability & Future Value"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and insert your DagsHub MLflow credentials:
```bash
cp .env.example .env
```
In `.env`:
```env
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repo_name>.mlflow
MLFLOW_TRACKING_USERNAME=<your_dagshub_username>
MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>
```

### 4. Launch the Streamlit Web Application
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

### 5. Run Automated End-to-End Tests
Execute the automated test suite covering unseen locality fallbacks, boundary stress tests, latency benchmarks, and live cloud audit logging:
```bash
python -m unittest -v tests/test_e2e_pipeline.py
```
*Expected Output: `Ran 4 tests in ~23s — OK` (p50 latency < 40ms, 100% cloud audit verification).*

---

## 🛡️ Enterprise Use Cases

1. **Individual Homebuyers & Retail Investors:** Unbiased, automated valuation tool to verify whether a seller's asking price is competitive relative to micro-market benchmarks before signing purchase agreements.
2. **Institutional Asset Managers & REITs:** Batch-evaluate thousands of potential property acquisitions simultaneously, automatically filtering for high-confidence buys with projected CAGRs exceeding hurdle rates.
3. **Mortgage Underwriters & Commercial Banks:** 5-year collateral stress-testing and loan-to-value (LTV) risk mitigation to guard against asset depreciation.
4. **Property Developers & Urban Planners:** Calibrate optimal unit sizes, pricing strategies, and amenity packages for new residential launches.

---

## 📄 License & Attribution

This project is licensed under the **MIT License**. Developed with the **Antigravity MLOps Framework**.
Dataset: 250,000 Pan-Indian Residential Listings across 42 Metros & Growth Corridors.

