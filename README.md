# Vantara Customer Intelligence Platform
### End-to-End Enterprise Machine Learning & Deep Learning Retention System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary
Vantara Customer Intelligence Platform is an applied machine learning and customer analytics engine built on real transactional retail data (**1,067,371 records**, Online Retail II, UCI ML Repository). The system transforms uncurated order logs into point-in-time customer feature tables to solve reactive retention, identifying at-risk high-value customers early, forecasting Customer Lifetime Value (CLV), generating behavioral segments, and providing explainable AI (XAI) outputs for business stakeholders.

---

## 🏗️ Architecture & Technical Highlights
* **Leakage-Free Feature Store**: Strict cutoff discipline separating historical observation window (`2009-12-01` to `2011-08-31`) from future churn evaluation window (`2011-09-01` to `2011-12-09`).
* **Multi-Model Benchmark**: Evaluated 6 classical ML classifiers, 1 PyTorch Deep Learning ANN, and Random Forest CLV Regressor.
* **Explainable AI (XAI)**: Integrated TreeSHAP explainer translating complex model weights into plain-language business drivers.
* **Serving Layer**: High-performance FastAPI REST API paired with an interactive multi-view Streamlit dashboard.

---

## 📊 Churn Prediction Benchmark (Test Set Evaluation)

| Model | Accuracy | Recall | Precision | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **0.7297** | **0.8169** | **0.7277** | **0.7697** | **0.7982** | **Production Champion** |
| **Deep Learning (ANN)** | 0.7230 | 0.7997 | 0.7268 | 0.7615 | 0.7987 | Deep Learning Benchmark |
| **Random Forest** | 0.7278 | 0.8204 | 0.7241 | 0.7692 | 0.7968 | Candidate |
| **LightGBM** | 0.7316 | 0.8256 | 0.7264 | 0.7728 | 0.7934 | Candidate |
| **Logistic Regression** | 0.7125 | 0.7893 | 0.7186 | 0.7523 | 0.7866 | Linear Baseline |
| **Support Vector Machine** | 0.7326 | 0.8117 | 0.7332 | 0.7705 | 0.7754 | Distance Baseline |
| **Decision Tree** | 0.7125 | 0.7841 | 0.7206 | 0.7510 | 0.7753 | Rule Baseline |

* **CLV Regression**: $R^2 = 0.6918$, $\text{MAE} = £416.62$, $\text{RMSE} = £1,483.78$

---

## 👥 Customer Behavioral Segments

| Segment | Customer Count | Avg Recency | Avg Frequency | Avg Historical Spend | Churn Rate | Strategic Marketing Focus |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **High-Value Champions (VIP)** | 978 | 40.0 days | 17.6 orders | £10,089.34 | 16.3% | VIP Concierge, Loyalty Rewards |
| **Loyal Regular Buyers** | 1,671 | 123.3 days | 4.9 orders | £1,385.66 | 45.6% | Cross-sell, Milestone Discounts |
| **At-Risk / Low-Frequency Buyers**| 2,585 | 317.7 days | 1.6 orders | £506.81 | 76.4% | Reactivation Campaigns, Price Drops |

---

## 📂 Repository Directory Layout
```text
customer-behavior-prediction/
├── api/
│   ├── main.py               # FastAPI application endpoints
│   └── schemas/customer.py   # Pydantic request & response contracts
├── config/config.yaml        # Pipeline paths and modeling hyperparameters
├── data/
│   ├── raw/                  # Source Online Retail II workbook
│   ├── interim/              # Sanitized parquet transaction logs
│   └── processed/            # Final customer feature store
├── frontend/dashboard.py     # Interactive Streamlit intelligence console
├── models_artifacts/         # Serialized models (.pkl, .pt), scaler, metrics
├── src/
│   ├── data/clean_data.py    # Raw transaction cleaning and deduplication
│   ├── features/build_features.py # RFM & leakage-free feature engineering
│   ├── models/train.py       # ML/DL training & benchmark comparison
│   ├── segmentation/cluster.py # K-Means & DBSCAN clustering
│   └── explainability/explain.py # SHAP global & local explainers
├── tests/test_features.py    # Pytest unit & leakage-prevention test suite
├── docker-compose.yml        # Multi-container orchestration config
├── Dockerfile                # Container build specification
├── requirements.txt          # Pinned runtime dependencies
└── README.md                 # System documentation

🚀 Quickstart & Execution
1. Local Environment Setup
Bash
git clone [https://github.com/akkii01/customer-behavior-prediction.git](https://github.com/akkii01/customer-behavior-prediction.git)
cd customer-behavior-prediction
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate
pip install -r requirements.txt

2. Run Data Pipeline & Training
Bash
python src/data/clean_data.py
python src/features/build_features.py
python src/models/train.py
python src/segmentation/cluster.py
python src/explainability/explain.py

3. Launch Applications
Bash
# Start FastAPI Service (Port 8000)
uvicorn api.main:app --reload

# Start Streamlit Dashboard (Port 8501)
streamlit run frontend/dashboard.py

4. Run Automated Tests
Bash
pytest tests/test_features.py