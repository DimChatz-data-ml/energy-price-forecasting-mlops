"""
Feature Engineering Pipeline (ELT: Transform Phase)
Reads: raw_day_ahead_prices, raw_load, raw_generation_long
Outputs: data/feature_matrix.parquet (ML-ready)
"""
import os
import logging
import polars as pl
from sqlalchemy import create_engine
from dotenv import load_dotenv
from datetime import date

# Load config
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)

def read_raw_data(engine):
    """Reads raw tables into Polars DataFrames using SQLAlchemy engine."""
    logging.info("📖 Reading raw tables from Postgres...")
    
    # ✅ FIXED: Use pl.read_database() instead of read_database_uri()
    #           and pass the engine object via 'connection' parameter
    prices = pl.read_database(
        query="SELECT country_code, period_start, price_eur_mwh FROM raw_day_ahead_prices",
        connection=engine,
    )
    load = pl.read_database(
        query="SELECT country_code, period_start, load_mw FROM raw_load",
        connection=engine,
    )
    gen_long = pl.read_database(
        query="SELECT country_code, period_start, fuel_type, gen_mw FROM raw_generation_long",
        connection=engine,
    )
    
    # Ensure consistent UTC timezone
    for df in [prices, load, gen_long]:
        df = df.with_columns(
            pl.col("period_start").cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
        )
        
    return prices, load, gen_long

def engineer_features(prices, load, gen_long):
    """Creates ML-ready features using Polars."""
    
    # 1. Pivot generation from long → wide
    logging.info("🔄 Pivoting generation data (long → wide)...")
    gen_wide = gen_long.pivot(
        on="fuel_type",
        index=["country_code", "period_start"],
        values="gen_mw",
        aggregate_function="sum"
    ).fill_null(0.0)
    
    # 2. Join datasets
    logging.info("🔗 Joining prices, load, and generation...")
    df = prices.join(load, on=["country_code", "period_start"], how="left")
    df = df.join(gen_wide, on=["country_code", "period_start"], how="left")
    
    # 3. Resample to hourly frequency & handle missing values
    logging.info("📊 Resampling to hourly frequency & filling gaps...")
    df = df.sort("period_start").group_by_dynamic(
        index_column="period_start",
        every="1h",
        by="country_code"
    ).agg(pl.exclude("country_code", "period_start").mean())
    
    # Forward-fill load, zero-fill generation
    df = df.with_columns([
        pl.col("load_mw").fill_null(strategy="forward").fill_null(0.0)
    ])
    
    # ✅ FIXED: Dynamic column selection for generation columns
    #           (avoids Polars version compatibility issues with "gen_*" wildcard)
    gen_cols = [c for c in df.columns if c.startswith("gen_")]
    if gen_cols:
        df = df.with_columns([pl.col(c).fill_null(0.0) for c in gen_cols])
    
    # 4. Calendar features
    logging.info("📅 Adding calendar features...")
    df = df.with_columns([
        pl.col("period_start").dt.hour().alias("hour"),
        pl.col("period_start").dt.weekday().alias("dayofweek"),  # 1=Mon, 7=Sun
        pl.col("period_start").dt.month().alias("month"),
        (pl.col("period_start").dt.weekday().is_in([6, 7])).cast(pl.Int8).alias("is_weekend"),
    ])
    
    # 5. Time-series features (lags & rolling stats per country)
    logging.info("📈 Creating time-series features...")
    # ✅ FIXED: Strict sorting before lags to prevent incorrect data transfer
    df = df.sort(["country_code", "period_start"])
    
    df = df.with_columns([
        pl.col("price_eur_mwh").shift(24).over("country_code").alias("lag_24h"),
        pl.col("price_eur_mwh").shift(168).over("country_code").alias("lag_168h"),
        pl.col("price_eur_mwh").rolling_mean(window_size=24).over("country_code").alias("price_mean_24h"),
        pl.col("price_eur_mwh").rolling_std(window_size=24).over("country_code").alias("price_std_24h"),
        pl.col("load_mw").rolling_mean(window_size=168).over("country_code").alias("load_mean_7d"),
    ])
    
    # 6. Regulatory flag (Iberian Exception for Spain: Jun 2022 - May 2023)
    df = df.with_columns(
        pl.when(
            (pl.col("country_code") == "ES") &
            (pl.col("period_start").dt.date().is_between(date(2022, 6, 1), date(2023, 5, 31)))
        ).then(1).otherwise(0).alias("es_price_cap_flag")
    )
    
    # 7. Drop initial rows with NaNs (due to lags/rolling windows + missing targets)
    df = df.drop_nulls(subset=["lag_168h", "lag_24h", "price_eur_mwh"])
    df = df.sort("country_code", "period_start")
    
    return df

def split_and_save(df, test_ratio=0.2):
    """Chronological train/test split per country (NO future leak)."""
    logging.info("✂️  Chronological train/test split per country...")
    train_dfs, test_dfs = [], []
    
    for country in df["country_code"].unique():
        country_df = df.filter(pl.col("country_code") == country)
        n = len(country_df)
        split_idx = int(n * (1 - test_ratio))
        train_dfs.append(country_df.slice(0, split_idx))
        test_dfs.append(country_df.slice(split_idx))
        
    train_df = pl.concat(train_dfs)
    test_df = pl.concat(test_dfs)
    
    # Save to Parquet
    paths = {
        "full": f"{DATA_DIR}/feature_matrix.parquet",
        "train": f"{DATA_DIR}/feature_matrix_train.parquet",
        "test": f"{DATA_DIR}/feature_matrix_test.parquet"
    }
    train_df.write_parquet(paths["train"])
    test_df.write_parquet(paths["test"])
    df.write_parquet(paths["full"])
    
    logging.info(f"💾 Saved: {len(train_df)} train, {len(test_df)} test, {len(df)} total rows")
    return train_df, test_df

def main():
    if not DB_URL:
        raise ValueError("❌ Missing DATABASE_URL in .env")
        
    engine = get_engine()
    prices, load, gen_long = read_raw_data(engine)
    
    df = engineer_features(prices, load, gen_long)
    logging.info(f"✅ Feature matrix ready: {df.shape[0]} rows × {df.shape[1]} cols")
    
    split_and_save(df)
    logging.info("🏁 Transformation pipeline completed!")

if __name__ == "__main__":
    main()