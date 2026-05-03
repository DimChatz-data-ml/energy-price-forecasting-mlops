import polars as pl
from pathlib import Path

# Εγγυημένο path προς το root του project
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def main():
    print("="*60)
    print("🔍 STARTING DATA AUDIT (Train/Val/Test Edition)")
    print(f"📂 Project Root: {BASE_DIR}")
    print(f"📂 Checking Data In: {DATA_DIR}")
    print("="*60)
    
    # 0️⃣ LOAD DATA
    files = {
        "train": DATA_DIR / "feature_matrix_train.parquet",
        "val": DATA_DIR / "feature_matrix_val.parquet",
        "test": DATA_DIR / "feature_matrix_test.parquet"
    }

    # Έλεγχος αν υπάρχουν τα αρχεία πριν τα διαβάσουμε
    for name, path in files.items():
        if not path.exists():
            print(f"❌ ERROR: Missing {name} file at {path}")
            print("💡 Tip: Run 'uv run src/transform_data.py' first.")
            return

    train = pl.read_parquet(files["train"])
    val = pl.read_parquet(files["val"])
    test = pl.read_parquet(files["test"])
    
    # Ενοποίηση για γενικούς ελέγχους (Schema, Nulls)
    df = pl.concat([train, val, test])

    # 1️⃣ SCHEMA & TYPES CHECK
    print("\n📐 1. SCHEMA CHECK")
    # Ορίζουμε τι περιμένουμε να δούμε
    expected_types = {
        "period_start": pl.Datetime("us", "UTC"),
        "price_eur_mwh": pl.Float64,
        "load_mw": pl.Float64,
        "hour": pl.Int8,
        "is_weekend": pl.Int8,
        "lag_24h": pl.Float64
    }
    
    for col, expected in expected_types.items():
        if col in df.columns:
            actual = df.schema.get(col)
            # Χρησιμοποιούμε base_type() για να μην κολλάμε σε λεπτομέρειες precision
            status = "✅" if actual == expected else "❌"
            print(f"  {status} {col}: {actual} (expected {expected})")
        else:
            print(f"  ❌ {col}: MISSING COLUMN!")
        
    # 2️⃣ NULL COUNTS (Critical Features)
    print("\n🕳️ 2. NULL COUNTS")
    critical_cols = ["price_eur_mwh", "lag_24h", "lag_168h", "price_mean_24h"]
    # Φιλτράρουμε μόνο όσες στήλες υπάρχουν όντως
    cols_to_check = [c for c in critical_cols if c in df.columns]
    
    nulls = df.select(cols_to_check).null_count()
    for col in cols_to_check:
        count = nulls[col][0]
        status = "✅" if count == 0 else "❌"
        print(f"  {status} {col}: {count} nulls")
        
    # 3️⃣ CHRONOLOGICAL INTEGRITY (Per Country)
    print("\n⏱️ 3. CHRONOLOGICAL ORDER")
    for c in df["country_code"].unique():
        sub = df.filter(pl.col("country_code") == c)
        is_sorted = sub["period_start"].is_sorted()
        status = "✅" if is_sorted else "❌"
        print(f"  {status} {c}: Sorted={is_sorted}")
        
    # 4️⃣ TARGET DISTRIBUTION
    print("\n📊 4. TARGET DISTRIBUTION (price_eur_mwh)")
    desc = df["price_eur_mwh"].describe()
    print(desc)
    
    # Έλεγχος για ακραίες τιμές (Sanity Check)
    if df["price_eur_mwh"].max() > 2000 or df["price_eur_mwh"].min() < -500:
        print("  ⚠️ WARNING: Extreme prices detected. Check your source data.")
    
    # 5️⃣ SPLIT INTEGRITY (Leak Detection)
    print("\n✂️ 5. TRAIN/VAL/TEST SPLIT INTEGRITY")
    leak_found = False
    for c in df["country_code"].unique():
        t_max = train.filter(pl.col("country_code") == c)["period_start"].max()
        v_min = val.filter(pl.col("country_code") == c)["period_start"].min()
        v_max = val.filter(pl.col("country_code") == c)["period_start"].max()
        te_min = test.filter(pl.col("country_code") == c)["period_start"].min()
        
        # Έλεγχος αν το train τελειώνει ΠΡΙΝ ξεκινήσει το val, κλπ.
        gap_1 = v_min > t_max if v_min and t_max else False
        gap_2 = te_min > v_max if te_min and v_max else False
        
        status = "✅" if (gap_1 and gap_2) else "❌ LEAK!"
        print(f"  {status} {c}: Train max ({t_max}) | Val min ({v_min}) | Test min ({te_min})")
        if not (gap_1 and gap_2): 
            leak_found = True
        
    print("\n" + "="*60)
    if leak_found:
        print("⛔ AUDIT FAILED: Data leak or overlap detected between splits.")
    else:
        print("✅ AUDIT PASSED: Data ready for ML training.")
    print("="*60)

if __name__ == "__main__":
    main()