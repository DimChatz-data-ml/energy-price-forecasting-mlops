import polars as pl

# Διάβασε το feature matrix
df = pl.read_parquet("data/feature_matrix.parquet")

# Βασικοί έλεγχοι
print(f"✅ Shape: {df.shape}")  # (~276k rows × 23 cols)
print(f"✅ Columns: {df.columns}")
print(f"✅ Nulls in target: {df['price_eur_mwh'].null_count()}")  # Πρέπει να είναι 0

# Έλεγξε chronological split (NO future leak!)
train = pl.read_parquet("data/feature_matrix_train.parquet")
test = pl.read_parquet("data/feature_matrix_test.parquet")

for country in df["country_code"].unique():
    train_max = train.filter(pl.col("country_code") == country)["period_start"].max()
    test_min = test.filter(pl.col("country_code") == country)["period_start"].min()
    print(f"✅ {country}: Train ends {train_max} | Test starts {test_min}")
    # Το test_min πρέπει να είναι >= train_max