import os
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import shap

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.helpers import setup_logger, load_config

logger = setup_logger("ExplainabilityPipeline")


def generate_plain_language_explanation(customer_id: int, churn_prob: float, top_features: list) -> str:
    """PRD Section 10: Translates technical SHAP values into business-friendly marketing reasoning."""
    risk_level = "HIGH RISK" if churn_prob >= 0.65 else ("LOW RISK" if churn_prob <= 0.35 else "MODERATE RISK")
    
    explanation = (
        f"Customer #{customer_id} is flagged as [{risk_level}] with a {churn_prob*100:.1f}% churn probability.\n"
        f"Key Business Drivers:\n"
    )
    for feat, val, impact in top_features:
        direction = "increasing churn risk" if impact > 0 else "encouraging customer retention"
        explanation += f"  - {feat} ({val:.2f}): Strongly {direction}.\n"
    return explanation


def run_explainability():
    config = load_config()
    processed_path = config["data"]["processed_features_path"]
    artifacts_dir = Path("models_artifacts")
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading production model, scaler, and customer feature store...")
    with open(artifacts_dir / "xgboost_churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(artifacts_dir / "features.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    df = pd.read_parquet(processed_path)
    X = df[feature_cols].copy()

    # 1. Initialize TreeSHAP Explainer (PRD Section 10)
    logger.info("Initializing TreeSHAP Explainer for XGBoost production model...")
    explainer = shap.TreeExplainer(model)
    
    # Save TreeSHAP explainer artifact for Dashboard & REST API
    with open(artifacts_dir / "shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)

    # 2. Compute Global Feature Importance
    logger.info("Computing SHAP values across representative customer sample (500 profiles)...")
    sample_X = X.sample(n=min(500, len(X)), random_state=42)
    shap_values = explainer(sample_X)

    # Mean absolute SHAP values per feature
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    global_importance = pd.DataFrame({
        "Feature": feature_cols,
        "Mean_SHAP_Impact": mean_abs_shap
    }).sort_values(by="Mean_SHAP_Impact", ascending=False)

    global_importance.to_csv(artifacts_dir / "shap_global_feature_importance.csv", index=False)

    print("\n" + "=" * 70)
    print("        GLOBAL MODEL EXPLAINABILITY (TOP CHURN PREDICTORS)")
    print("=" * 70)
    print(global_importance.to_string(index=False))
    print("=" * 70)

    # 3. Worked Individual Customer Local Explanations
    logger.info("Generating worked local explainability templates for representative cases...")
    df["churn_probability"] = model.predict_proba(X)[:, 1]

    # Find one high risk and one low risk customer
    high_risk_idx = df["churn_probability"].idxmax()
    low_risk_idx = df["churn_probability"].idxmin()

    for label, idx in [("HIGH RISK CASE", high_risk_idx), ("LOYAL / LOW RISK CASE", low_risk_idx)]:
        cust_row = df.loc[idx]
        cust_id = int(cust_row["customer_id"])
        prob = float(cust_row["churn_probability"])
        
        single_x = X.loc[[idx]]
        cust_shap = explainer(single_x).values[0]

        top_indices = np.argsort(np.abs(cust_shap))[-3:][::-1]
        top_feats = [
            (feature_cols[i], float(single_x.iloc[0, i]), float(cust_shap[i]))
            for i in top_indices
        ]

        text_summary = generate_plain_language_explanation(cust_id, prob, top_feats)
        print(f"\n--- [{label}] ---")
        print(text_summary)

    logger.info("Explainability phase completed. Explainer artifacts serialized.")


if __name__ == "__main__":
    run_explainability()