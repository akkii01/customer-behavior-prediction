import os
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

# Modeling & Evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error
)
import xgboost as xgb
import lightgbm as lgb

# Deep Learning (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.helpers import setup_logger, load_config

logger = setup_logger("ModelTrainingPipeline")


class ChurnANN(nn.Module):
    """PRD Section 8.2: Feed-forward ANN with Batch Normalization and Dropout."""
    def __init__(self, input_dim: int):
        super(ChurnANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def prepare_datasets(df: pd.DataFrame):
    """Selects numerical features, splits into Train/Test, and scales appropriately."""
    feature_cols = [
        "frequency", "total_items", "monetary_value", "recency", "tenure_days",
        "avg_order_value", "avg_basket_size", "purchase_velocity",
        "avg_days_between", "std_days_between", "engagement_score"
    ]
    
    X = df[feature_cols].copy()
    y_churn = df["is_churn"].values
    y_clv = df["future_clv"].values

    # Stratified Train/Test Split (80% train, 20% held-out test)
    X_train, X_test, y_train_churn, y_test_churn, y_train_clv, y_test_clv = train_test_split(
        X, y_churn, y_clv, test_size=0.20, random_state=42, stratify=y_churn
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        feature_cols, scaler,
        X_train, X_test, X_train_scaled, X_test_scaled,
        y_train_churn, y_test_churn, y_train_clv, y_test_clv
    )


def train_and_evaluate_models():
    config = load_config()
    processed_path = config["data"]["processed_features_path"]
    artifacts_dir = Path("models_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading feature store from: {processed_path}")
    df = pd.read_parquet(processed_path)

    (
        feature_cols, scaler,
        X_train, X_test, X_train_scaled, X_test_scaled,
        y_train_churn, y_test_churn, y_train_clv, y_test_clv
    ) = prepare_datasets(df)

    # Save feature names & scaler for API inference
    with open(artifacts_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(artifacts_dir / "features.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    results = []

    # 1. Classical ML Classifiers
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
        "Decision Tree": (DecisionTreeClassifier(max_depth=6, random_state=42), False),
        "Random Forest": (RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42), False),
        "XGBoost": (xgb.XGBClassifier(n_estimators=120, max_depth=5, learning_rate=0.05, random_state=42, eval_metric="logloss"), False),
        "LightGBM": (lgb.LGBMClassifier(n_estimators=120, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1), False),
        "Support Vector Machine": (SVC(probability=True, kernel="rbf", random_state=42), True)
    }

    trained_models = {}

    for name, (model, requires_scale) in models.items():
        logger.info(f"Training {name}...")
        X_tr = X_train_scaled if requires_scale else X_train
        X_te = X_test_scaled if requires_scale else X_test

        model.fit(X_tr, y_train_churn)
        preds = model.predict(X_te)
        probs = model.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_test_churn, preds)
        rec = recall_score(y_test_churn, preds)
        prec = precision_score(y_test_churn, preds)
        f1 = f1_score(y_test_churn, preds)
        auc = roc_auc_score(y_test_churn, probs)

        trained_models[name] = model
        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Recall": round(rec, 4),
            "Precision": round(prec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        })

    # 2. PyTorch Feed-Forward ANN
    logger.info("Training Deep Learning ANN Model...")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_churn, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

    ann_model = ChurnANN(input_dim=len(feature_cols)).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(ann_model.parameters(), lr=0.005, weight_decay=1e-4)

    ann_model.train()
    for epoch in range(40):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = ann_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    ann_model.eval()
    with torch.no_grad():
        ann_probs = ann_model(X_test_t.to(device)).cpu().numpy().flatten()
        ann_preds = (ann_probs >= 0.5).astype(int)

    results.append({
        "Model": "Deep Learning (ANN)",
        "Accuracy": round(accuracy_score(y_test_churn, ann_preds), 4),
        "Recall": round(recall_score(y_test_churn, ann_preds), 4),
        "Precision": round(precision_score(y_test_churn, ann_preds), 4),
        "F1-Score": round(f1_score(y_test_churn, ann_preds), 4),
        "ROC-AUC": round(roc_auc_score(y_test_churn, ann_probs), 4)
    })

    # Save Best Production Model (XGBoost) & ANN
    with open(artifacts_dir / "xgboost_churn_model.pkl", "wb") as f:
        pickle.dump(trained_models["XGBoost"], f)
    torch.save(ann_model.state_dict(), artifacts_dir / "ann_churn_model.pt")

    # 3. CLV Regression Model (PRD Section 8 & 11)
    logger.info("Training CLV Regression Model (Random Forest Regressor)...")
    clv_regressor = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
    clv_regressor.fit(X_train, y_train_clv)
    clv_preds = clv_regressor.predict(X_test)

    r2 = r2_score(y_test_clv, clv_preds)
    mae = mean_absolute_error(y_test_clv, clv_preds)
    rmse = np.sqrt(mean_squared_error(y_test_clv, clv_preds))

    with open(artifacts_dir / "clv_regressor.pkl", "wb") as f:
        pickle.dump(clv_regressor, f)

    # 4. Display & Persist Results Table (PRD Section 11)
    results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    results_df.to_csv(artifacts_dir / "model_comparison_metrics.csv", index=False)

    print("\n" + "=" * 65)
    print("      VANTARA CHURN PREDICTION BENCHMARK TABLE (TEST SET)")
    print("=" * 65)
    print(results_df.to_string(index=False))
    print("=" * 65)
    print(f"CLV Regression -> R2 Score: {r2:.4f} | MAE: £{mae:.2f} | RMSE: £{rmse:.2f}")
    print("=" * 65 + "\n")
    logger.info("Model training & serialization complete. Artifacts saved in models_artifacts/.")


if __name__ == "__main__":
    train_and_evaluate_models()