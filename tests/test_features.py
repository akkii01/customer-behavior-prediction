import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_feature_data():
    processed_path = Path("data/processed/customer_features.parquet")
    if not processed_path.exists():
        pytest.skip("Processed feature dataset not found.")
    return pd.read_parquet(processed_path)


def test_no_null_critical_features(sample_feature_data):
    """PRD Section 6.2: Ensure zero null values in critical model features."""
    critical_cols = [
        "customer_id", "recency", "frequency", "monetary_value",
        "engagement_score", "is_churn", "future_clv"
    ]
    for col in critical_cols:
        assert sample_feature_data[col].isnull().sum() == 0, f"Null values detected in {col}"


def test_target_leakage_absence(sample_feature_data):
    """PRD Section 7: Verify point-in-time target isolation (future_clv vs is_churn alignment)."""
    # If customer had 0 future CLV, they must be flagged as churn = 1
    zero_spenders = sample_feature_data[sample_feature_data["future_clv"] == 0]
    assert (zero_spenders["is_churn"] == 1).all(), "Target leakage detected: zero future CLV customer not marked as churn."

    # If customer spent in future window, churn must be 0
    active_spenders = sample_feature_data[sample_feature_data["future_clv"] > 0]
    assert (active_spenders["is_churn"] == 0).all(), "Target leakage detected: positive future CLV marked as churn."


def test_engagement_score_bounds(sample_feature_data):
    """PRD Section 7: Verify composite engagement score is strictly bounded [0, 1]."""
    scores = sample_feature_data["engagement_score"]
    assert (scores >= 0.0).all() and (scores <= 1.0).all(), "Engagement score exceeds expected [0, 1] range."