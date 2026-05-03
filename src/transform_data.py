"""
Feature Engineering Pipeline (ELT: Transform Phase)
Reads: raw_day_ahead_prices, raw_load, raw_generation_long
Outputs: data/feature_matrix_{train,val,test,full}.parquet (ML-ready)
"""
import os
import logging
import polars as pl
from pathlib import Path
from sqlalchemy import create_engine
from dotenv import load_dotenv
from datetime import date

# --- ΔΥΝΑΜΙΚΗ ΔΙΑΧΕΙΡΙΣΗ PATHS ---
# Βρίσκουμε το root του project (energy-mlops) με βάση τη θέση αυτού του αρχείου
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Δημιουργία του φακέλου data στο root αν δεν υπάρχει
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Φόρτωση ρυθμίσεων
load_dotenv(BASE_DIR / ".env")
DB_URL = os.getenv("DATABASE_URL")

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)

def read_raw_data(engine):
    """Reads raw tables into Polars DataFrames."""
    logging.info("📖 Reading raw tables from Postgres...")
    
    # Χρήση connection αντί για engine απευθείας για αποφυγή TypeError σε κάποιες εκδόσεις
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
    
    # 3. Resample & Handle Missing
    logging.info("📊 Resampling to hourly frequency & filling gaps...")
    df = df.sort("period_start").group_by_dynamic(
        index_column="period_start",
        every="1h",
        group_by="country_code"
    ).agg(pl.exclude("country_code", "period_start").mean())
    
    df = df.with_columns([
        pl.col("load_mw").fill_null(strategy="forward").fill_null(0.0)
    ])
    
    # 4. Calendar features
    logging.info("📅 Adding calendar features...")
    df = df.with_columns([
        pl.col("period_start").dt.hour().cast(pl.Int8).alias("hour"),
        pl.col("period_start").dt.weekday().cast(pl.Int8).alias("dayofweek"),
        pl.col("period_start").dt.month().cast(pl.Int8).alias("month"),
        (pl.col("period_start").dt.weekday().is_in([6, 7])).cast(pl.Int8).alias("is_weekend"),
    ])
    
    # 5. Time-series features
    logging.info("📈 Creating time-series features...")
    df = df.sort(["country_code", "period_start"])
    
    df = df.with_columns([
        pl.col("price_eur_mwh").shift(24).over("country_code").alias("lag_24h"),
        pl.col("price_eur_mwh").shift(168).over("country_code").alias("lag_168h"),
        pl.col("price_eur_mwh").rolling_mean(window_size=24).over("country_code").alias("price_mean_24h"),
        pl.col("load_mw").rolling_mean(window_size=168).over("country_code").alias("load_mean_7d"),
    ])
    
    # 6. Iberian Exception Flag
    df = df.with_columns(
        pl.when(
            (pl.col("country_code") == "ES") &
            (pl.col("period_start").dt.date().is_between(date(2022, 6, 1), date(2023, 5, 31)))
        ).then(1).otherwise(0).cast(pl.Int8).alias("es_price_cap_flag")
    )
    
    df = df.drop_nulls(subset=["lag_168h", "lag_24h", "price_eur_mwh"])
    return df.sort("country_code", "period_start")

def split_and_save(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """Chronological train/val/test split using absolute paths."""
    
    logging.info(f"✂️ Splitting data per country to: {DATA_DIR}")
    train_dfs, val_dfs, test_dfs = [], [], []
    
    for country in df["country_code"].unique():
        country_df = df.filter(pl.col("country_code") == country).sort("period_start")
        n = len(country_df)
        
        t_idx = int(n * train_ratio)
        v_idx = int(n * (train_ratio + val_ratio))
        
        train_dfs.append(country_df.slice(0, t_idx))
        val_dfs.append(country_df.slice(t_idx, v_idx - t_idx))
        test_dfs.append(country_df.slice(v_idx))
        
    # Concat and Write
    sets = {
        "full": (df, "feature_matrix.parquet"),
        "train": (pl.concat(train_dfs), "feature_matrix_train.parquet"),
        "val": (pl.concat(val_dfs), "feature_matrix_val.parquet"),
        "test": (pl.concat(test_dfs), "feature_matrix_test.parquet")
    }

    for name, (data, filename) in sets.items():
        save_path = DATA_DIR / filename
        data.write_parquet(save_path)
        logging.info(f"💾 Saved {name} set to {save_path} ({len(data)} rows)")

def main():
    if not DB_URL:
        raise ValueError("❌ DATABASE_URL not found. Check your .env file.")
        
    engine = get_engine()
    prices, load, gen_long = read_raw_data(engine)
    df = engineer_features(prices, load, gen_long)
    
    logging.info(f"✅ Feature matrix ready: {df.shape}")
    split_and_save(df)
    logging.info("🏁 Pipeline Finished!")

if __name__ == "__main__":
    main()