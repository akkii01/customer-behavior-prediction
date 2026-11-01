import os
import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.helpers import setup_logger, load_config

logger = setup_logger("DataCleaningPipeline")


def load_and_combine_raw_data(excel_path: str) -> pd.DataFrame:
    """Loads both sheets from Online Retail II Excel workbook and concatenates them."""
    logger.info(f"Loading raw workbook from: {excel_path} (This may take a moment...)")

    if not Path(excel_path).exists():
        raise FileNotFoundError(f"Source raw dataset missing at: {excel_path}")

    # Read both transactional sheets defined in PRD Section 5.1
    df_2009 = pd.read_excel(excel_path, sheet_name="Year 2009-2010")
    logger.info(f"Sheet 2009-2010 loaded: {len(df_2009):,} records")

    df_2010 = pd.read_excel(excel_path, sheet_name="Year 2010-2011")
    logger.info(f"Sheet 2010-2011 loaded: {len(df_2010):,} records")

    combined_df = pd.concat([df_2009, df_2010], ignore_index=True)
    logger.info(f"Total raw transactions merged: {len(combined_df):,} records")
    return combined_df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans transaction logs following PRD Section 6 requirements."""
    logger.info("Starting data sanitation & validation checks...")

    # Standardize column naming
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]

    column_mapping = {
        "customer_id": "customer_id",
        "invoicedate": "invoice_date",
        "stockcode": "stock_code",
    }
    df = df.rename(columns=column_mapping)

    initial_count = len(df)

    # 1. PRD Section 5.3: Filter out transactions without valid Customer ID
    df = df.dropna(subset=["customer_id"]).copy()
    df["customer_id"] = df["customer_id"].astype("int64")
    logger.info(f"Filtered records with missing customer_id. Remaining: {len(df):,}")

    # 2. PRD Section 6.2: Deduplicate exact duplicate invoice rows
    df = df.drop_duplicates()
    logger.info(f"Deduplicated identical records. Remaining: {len(df):,}")

    # 3. Handle cancellations and negative values
    df["invoice"] = df["invoice"].astype(str)
    df["is_cancelled"] = df["invoice"].str.startswith("C") | (df["quantity"] <= 0)

    # Clean positive retail transactions for modeling
    clean_df = df[
        (~df["is_cancelled"]) & 
        (df["quantity"] > 0) & 
        (df["price"] > 0)
    ].copy()

    # 4. Filter administrative stock codes (Postage, Manual fees, etc.)
    non_product_codes = ["POST", "D", "M", "PADS", "DOT", "CRUK", "BANK CHARGES"]
    clean_df = clean_df[~clean_df["stock_code"].astype(str).str.upper().isin(non_product_codes)]

    # 5. Fix type inconsistencies for Parquet serialization
    clean_df["stock_code"] = clean_df["stock_code"].astype(str)
    clean_df["description"] = clean_df["description"].fillna("Unknown").astype(str)
    clean_df["country"] = clean_df["country"].fillna("Unknown").astype(str)
    clean_df["invoice"] = clean_df["invoice"].astype(str)

    # Compute Total Transaction Value
    clean_df["total_amount"] = (clean_df["quantity"] * clean_df["price"]).astype(float)
    clean_df["invoice_date"] = pd.to_datetime(clean_df["invoice_date"])

    clean_df = clean_df.sort_values(by="invoice_date").reset_index(drop=True)

    logger.info(
        f"Data cleaning finished: Retained {len(clean_df):,} valid transactions "
        f"from {initial_count:,} raw records."
    )
    return clean_df


def run_pipeline():
    config = load_config()
    raw_path = config["data"]["raw_path"]
    interim_path = config["data"]["interim_path"]

    os.makedirs(os.path.dirname(interim_path), exist_ok=True)

    raw_df = load_and_combine_raw_data(raw_path)
    cleaned_df = clean_transactions(raw_df)

    logger.info(f"Writing cleaned transactions to interim storage: {interim_path}")
    cleaned_df.to_parquet(interim_path, index=False)
    logger.info("Phase 1 Data Ingestion & Cleaning pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()