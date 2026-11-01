import os
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.schemas.customer import CustomerPredictionRequest, CustomerPredictionResponse

app = FastAPI(
    title="Vantara Customer Intelligence API",
    description="Production REST API for Customer Churn Prediction, CLV Forecasting, and Segmentation.",
    version="1.0.0"
)

# Load serialized model artifacts
ARTIFACTS_DIR = Path("models_artifacts")
DATA_PATH = Path("data/processed/customer_features.parquet")

try:
    with open(ARTIFACTS_DIR / "xgboost_churn_model.pkl", "rb") as f:
        churn_model = pickle.load(f)
    with open(ARTIFACTS_DIR / "clv_regressor.pkl", "rb") as f:
        clv_model = pickle.load(f)
    with open(ARTIFACTS_DIR / "features.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    with open(ARTIFACTS_DIR / "shap_explainer.pkl", "rb") as f:
        shap_explainer = pickle.load(f)
    customer_db = pd.read_parquet(DATA_PATH) if DATA_PATH.exists() else None
except Exception as e:
    print(f"Warning: Model artifacts loading exception: {e}")
    churn_model, clv_model, feature_cols, shap_explainer, customer_db = None, None, None, None, None


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "online",
        "service": "Vantara Customer Intelligence Platform",
        "models_loaded": churn_model is not None and clv_model is not None
    }


@app.get("/model-metadata", tags=["Model Governance"])
def get_metadata():
    metrics_path = ARTIFACTS_DIR / "model_comparison_metrics.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        metrics = metrics_df.to_dict(orient="records")
    else:
        metrics = []

    return {
        "production_churn_model": "XGBoost Classifier",
        "clv_regression_model": "Random Forest Regressor",
        "features": feature_cols,
        "benchmark_metrics": metrics
    }


@app.post("/predict/churn", response_model=CustomerPredictionResponse, tags=["Inference"])
def predict_churn(request: CustomerPredictionRequest):
    if churn_model is None or clv_model is None:
        raise HTTPException(status_code=500, detail="Models are not loaded.")

    data_dict = request.dict()
    input_vector = [data_dict[col] for col in feature_cols]
    input_df = pd.DataFrame([input_vector], columns=feature_cols)

    churn_prob = float(churn_model.predict_proba(input_df)[0, 1])
    churn_pred = int(churn_prob >= 0.5)
    risk_level = "HIGH RISK" if churn_prob >= 0.65 else ("LOW RISK" if churn_prob <= 0.35 else "MODERATE RISK")
    predicted_clv = float(clv_model.predict(input_df)[0])

    # Compute SHAP explanation on the fly
    shap_vals = shap_explainer(input_df).values[0]
    top_indices = np.argsort(np.abs(shap_vals))[-3:][::-1]
    top_drivers = [
        {
            "feature": feature_cols[i],
            "value": round(float(input_vector[i]), 2),
            "shap_impact": round(float(shap_vals[i]), 4),
            "effect": "Increases Churn Risk" if shap_vals[i] > 0 else "Supports Retention"
        }
        for i in top_indices
    ]

    return CustomerPredictionResponse(
        churn_probability=round(churn_prob, 4),
        churn_prediction=churn_pred,
        predicted_risk_level=risk_level,
        predicted_future_clv=round(max(0.0, predicted_clv), 2),
        top_churn_drivers=top_drivers
    )


@app.get("/customers/{customer_id}", tags=["Customer Intelligence"])
def get_customer_profile(customer_id: int):
    if customer_db is None:
        raise HTTPException(status_code=500, detail="Customer store unavailable.")

    record = customer_db[customer_db["customer_id"] == customer_id]
    if record.empty:
        raise HTTPException(status_code=404, detail=f"Customer ID {customer_id} not found.")

    res = record.iloc[0].to_dict()
    for k, v in res.items():
        if pd.isna(v):
            res[k] = None
        elif isinstance(v, (np.integer, np.int64)):
            res[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            res[k] = round(float(v), 2)

    return res