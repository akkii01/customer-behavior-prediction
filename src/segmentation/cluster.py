import os
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.helpers import setup_logger, load_config

logger = setup_logger("CustomerSegmentationPipeline")


def run_segmentation():
    config = load_config()
    processed_path = config["data"]["processed_features_path"]
    artifacts_dir = Path("models_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading feature store for clustering: {processed_path}")
    df = pd.read_parquet(processed_path)

    # Core RFM + Engagement features for behavioral clustering
    cluster_features = ["recency", "frequency", "monetary_value", "avg_order_value", "engagement_score"]
    X = df[cluster_features].copy()

    # Log-transform skewed monetary/frequency features for stable K-Means spherical boundaries
    X_log = np.log1p(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    # 1. K-Means Clustering (k=4 standard retail behavioral segments)
    logger.info("Executing K-Means clustering (k=4)...")
    kmeans = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
    df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

    # Evaluate K-Means
    km_sil = silhouette_score(X_scaled, df["kmeans_cluster"])
    km_db = davies_bouldin_score(X_scaled, df["kmeans_cluster"])
    logger.info(f"K-Means Evaluation -> Silhouette Score: {km_sil:.4f} | Davies-Bouldin: {km_db:.4f}")

    # 2. DBSCAN Density Clustering (PRD Section 9 validation)
    logger.info("Executing DBSCAN density check...")
    dbscan = DBSCAN(eps=1.2, min_samples=15)
    df["dbscan_cluster"] = dbscan.fit_predict(X_scaled)
    noise_count = (df["dbscan_cluster"] == -1).sum()
    logger.info(f"DBSCAN identified {noise_count:,} outlier customers out of {len(df):,}")

    # 3. Dynamic Business Persona Profiling (PRD Section 9)
    # Map cluster centroids to commercial names based on Recency & Monetary traits
    cluster_summary = df.groupby("kmeans_cluster").agg(
        recency_mean=("recency", "mean"),
        frequency_mean=("frequency", "mean"),
        monetary_mean=("monetary_value", "mean"),
        avg_order_mean=("avg_order_value", "mean"),
        churn_rate=("is_churn", "mean"),
        customer_count=("customer_id", "count")
    ).reset_index()

    # Heuristic business naming based on RFM centroid metrics
    persona_mapping = {}
    for _, row in cluster_summary.iterrows():
        c = int(row["kmeans_cluster"])
        m = row["monetary_mean"]
        r = row["recency_mean"]
        f = row["frequency_mean"]

        if m >= cluster_summary["monetary_mean"].quantile(0.75) or f >= cluster_summary["frequency_mean"].quantile(0.75):
            if r <= cluster_summary["recency_mean"].median():
                persona_mapping[c] = "High-Value Champions (VIP)"
            else:
                persona_mapping[c] = "Lapsed VIPs (High Retention Priority)"
        else:
            if r <= cluster_summary["recency_mean"].median():
                persona_mapping[c] = "Loyal Regular Buyers"
            else:
                persona_mapping[c] = "At-Risk / Low-Frequency Buyers"

    df["customer_segment"] = df["kmeans_cluster"].map(persona_mapping)

    # Serialize clustering artifacts
    with open(artifacts_dir / "kmeans_cluster_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    with open(artifacts_dir / "cluster_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(artifacts_dir / "persona_mapping.pkl", "wb") as f:
        pickle.dump(persona_mapping, f)

    # Overwrite processed parquet with segment tags attached
    df.to_parquet(processed_path, index=False)
    logger.info(f"Segment-enriched dataset updated at: {processed_path}")

    # Print business profile summary
    print("\n" + "=" * 80)
    print("               VANTARA CUSTOMER BEHAVIORAL SEGMENTS SUMMARY")
    print("=" * 80)
    final_summary = df.groupby("customer_segment").agg(
        Count=("customer_id", "count"),
        Avg_Recency_Days=("recency", lambda x: round(x.mean(), 1)),
        Avg_Frequency=("frequency", lambda x: round(x.mean(), 1)),
        Avg_Spend_GBP=("monetary_value", lambda x: f"£{x.mean():,.2f}"),
        Avg_CLV_GBP=("future_clv", lambda x: f"£{x.mean():,.2f}"),
        Churn_Rate=("is_churn", lambda x: f"{x.mean()*100:.1f}%")
    ).reset_index()
    print(final_summary.to_string(index=False))
    print("=" * 80 + "\n")
    logger.info("Customer segmentation & profiling phase completed successfully.")


if __name__ == "__main__":
    run_segmentation()