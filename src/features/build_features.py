import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.helpers import setup_logger, load_config

logger = setup_logger("FeatureEngineeringPipeline")


def engineer_customer_features(
    interim_path: str, cutoff_date_str: str = "2011-09-01"
) -> pd.DataFrame:
    """Transforms transaction logs into a customer-level analytical feature store

    with strict point-in-time cutoff to prevent target leakage (PRD Section 7).
    """
    logger.info(f"Loading cleaned transactions from: {interim_path}")
    df = pd.read_parquet(interim_path)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    cutoff_date = pd.to_datetime(cutoff_date_str)
    logger.info(f"Applying point-in-time feature cutoff date: {cutoff_date_str}")

    # Split dataset into historical observation window and future evaluation window
    historical_df = df[df["invoice_date"] < cutoff_date].copy()
    future_df = df[df["invoice_date"] >= cutoff_date].copy()

    logger.info(
        f"Historical window: {len(historical_df):,} records | Future window: {len(future_df):,} records"
    )

    # 1. Base Customer Aggregations in Observation Window
    logger.info("Computing customer-level behavioral and RFM metrics...")
    
    # Recency reference date is the cutoff date itself
    customers = historical_df.groupby("customer_id").agg(
        first_purchase=("invoice_date", "min"),
        last_purchase=("invoice_date", "max"),
        frequency=("invoice", "nunique"),
        total_items=("quantity", "sum"),
        monetary_value=("total_amount", "sum"),
        country=("country", "first"),
    ).reset_index()

    # Recency: Days from last purchase to cutoff date
    customers["recency"] = (cutoff_date - customers["last_purchase"]).dt.days
    
    # Customer Tenure: Days between first purchase and cutoff date
    customers["tenure_days"] = (cutoff_date - customers["first_purchase"]).dt.days
    customers["tenure_days"] = customers["tenure_days"].replace(0, 1)

    # Average Order Value (AOV) and Basket Size
    customers["avg_order_value"] = (
        customers["monetary_value"] / customers["frequency"]
    ).round(2)
    customers["avg_basket_size"] = (
        customers["total_items"] / customers["frequency"]
    ).round(2)

    # Purchase Velocity (Orders per 30 days active)
    customers["purchase_velocity"] = (
        customers["frequency"] / (customers["tenure_days"] / 30.0)
    ).round(4)

    # 2. Inter-purchase Time Variance (Regularity)
    logger.info("Calculating purchase cadence variance...")
    order_dates = (
        historical_df[["customer_id", "invoice_date"]]
        .drop_duplicates()
        .sort_values(by=["customer_id", "invoice_date"])
    )
    order_dates["prev_date"] = order_dates.groupby("customer_id")["invoice_date"].shift(1)
    order_dates["days_between"] = (
        order_dates["invoice_date"] - order_dates["prev_date"]
    ).dt.days

    interval_stats = order_dates.groupby("customer_id")["days_between"].agg(
        avg_days_between="mean",
        std_days_between="std"
    ).reset_index()
    
    interval_stats["avg_days_between"] = interval_stats["avg_days_between"].fillna(0)
    interval_stats["std_days_between"] = interval_stats["std_days_between"].fillna(0)

    customers = customers.merge(interval_stats, on="customer_id", how="left")

    # 3. Composite Engagement Score (PRD Section 7)
    # Scaled ranks for Recency (inverted), Frequency, and Monetary
    r_rank = 1.0 - (customers["recency"].rank(pct=True))
    f_rank = customers["frequency"].rank(pct=True)
    m_rank = customers["monetary_value"].rank(pct=True)

    customers["engagement_score"] = (
        (0.35 * r_rank) + (0.35 * f_rank) + (0.30 * m_rank)
    ).round(4)

    # 4. Construct Future Target Variables (Ground Truth)
    logger.info("Labeling churn and future customer lifetime value targets...")
    future_spenders = future_df.groupby("customer_id")["total_amount"].sum().reset_index()
    future_spenders.rename(columns={"total_amount": "future_clv"}, inplace=True)

    # Merge with historical customers
    features_df = customers.merge(future_spenders, on="customer_id", how="left")
    
    # If customer spent > 0 in future window, churn = 0, else churn = 1
    features_df["future_clv"] = features_df["future_clv"].fillna(0.0).round(2)
    features_df["is_churn"] = np.where(features_df["future_clv"] == 0.0, 1, 0)

    # Drop temporary datetime reference columns
    features_df.drop(columns=["first_purchase", "last_purchase"], inplace=True)

    churn_rate = features_df["is_churn"].mean() * 100
    logger.info(
        f"Feature store ready: {len(features_df):,} customers processed | Churn Rate: {churn_rate:.2f}%"
    )
    return features_df


def run_pipeline():
    config = load_config()
    interim_path = config["data"]["interim_path"]
    processed_path = config["data"]["processed_features_path"]
    cutoff_date = config["features"].get("recency_cutoff_date", "2011-09-01")

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    features_df = engineer_customer_features(interim_path, cutoff_date)

    logger.info(f"Saving feature table to: {processed_path}")
    features_df.to_parquet(processed_path, index=False)
    logger.info("Phase 2 Feature Engineering pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()