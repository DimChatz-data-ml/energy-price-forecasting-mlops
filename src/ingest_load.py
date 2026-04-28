"""
Fetch Load Data from ENTSO-E
Saves to: raw_load (Postgres)
Pattern: Chunked fetch → Validation → Transactional Upsert
"""
import os
import logging
import time
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from entsoe import EntsoePandasClient
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load configuration
load_dotenv()
API_KEY = os.getenv("ENTSOE_API_KEY")
DB_URL = os.getenv("DATABASE_URL")

# Target countries and date range
COUNTRIES = ["GR", "DE_LU", "FR", "ES", "PL"]
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)

def ensure_raw_load_table(engine):
    """Creates the raw_load table with PRIMARY KEY for idempotency."""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw_load (
                country_code VARCHAR(5) NOT NULL,
                period_start TIMESTAMPTZ NOT NULL,
                load_mw DECIMAL(10,2),
                ingestion_timestamp TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (country_code, period_start)
            );
        """))
        conn.commit()
        logging.info("✅ Raw load table ensured with PK constraint.")

def fetch_monthly_load_chunk(client, country_code, start_date, end_date):
    """Fetches load data for one country, one month at a time."""
    try:
        df = client.query_load(country_code, start=start_date, end=end_date)
        if df is not None and not df.empty:
            df = df.reset_index()
            df.columns = ['period_start', 'load_mw']
            df['country_code'] = country_code
            df['period_start'] = pd.to_datetime(df['period_start']).dt.tz_convert('UTC')
            return df
    except Exception as e:
        logging.warning(f"⚠️ Failed to fetch load {country_code} for {start_date.strftime('%Y-%m')}: {e}")
    return pd.DataFrame()

def validate_load_data(df, country_code):
    """Basic validation: remove duplicates, check reasonable load bounds."""
    original_len = len(df)
    df = df.drop_duplicates(subset=['country_code', 'period_start'])
    
    # Load should be positive and under realistic bounds (e.g. < 100 GW for large countries)
    df = df[df['load_mw'].between(0, 100_000)]  # Up to 100 GW
    
    if len(df) != original_len:
        logging.info(f"🧹 Cleaned {original_len - len(df)} invalid entries for {country_code}")
    return df

def main():
    if not API_KEY or not DB_URL:
        raise ValueError("❌ Missing API key or database URL in .env")
    
    client = EntsoePandasClient(api_key=API_KEY)
    engine = get_engine()
    
    # Ensure table exists
    ensure_raw_load_table(engine)
    
    for country in COUNTRIES:
        logging.info(f"🚀 Fetching load data for {country}...")
        start = pd.Timestamp(START_DATE, tz='UTC')
        end_limit = pd.Timestamp(END_DATE, tz='UTC')
        
        country_data = []
        while start < end_limit:
            chunk_end = min(start + relativedelta(months=1), end_limit)
            chunk = fetch_monthly_load_chunk(client, country, start, chunk_end)
            
            if not chunk.empty:
                chunk = validate_load_data(chunk, country)
                country_data.append(chunk)
            
            # Rate limiting: Give API a break
            time.sleep(0.5)
            start = chunk_end

        # Safe save per country with transactional upsert
        if country_data:
            country_df = pd.concat(country_data, ignore_index=True)
            country_df = country_df.drop_duplicates(subset=['country_code', 'period_start'])
            
            logging.info(f"💾 Saving {len(country_df)} load rows for {country} to DB...")
            
            try:
                with engine.begin() as conn:
                    # Create temp table for safe upsert
                    conn.execute(text("CREATE TEMP TABLE temp_load AS SELECT * FROM raw_load WITH NO DATA"))
                    country_df.to_sql('temp_load', conn, if_exists='append', index=False)
                    
                    # Upsert: Insert if new, ignore if exists
                    upsert_query = text("""
                        INSERT INTO raw_load (country_code, period_start, load_mw)
                        SELECT country_code, period_start, load_mw FROM temp_load
                        ON CONFLICT (country_code, period_start) 
                        DO NOTHING;
                    """)
                    conn.execute(upsert_query)
                    conn.execute(text("DROP TABLE temp_load"))
                    
                logging.info(f"✅ Load data for {country} saved successfully!")
            except Exception as e:
                logging.error(f"❌ Error saving load data for {country}: {e}")
        else:
            logging.warning(f"⏭️ No load data found for {country}.")

    logging.info("🏁 Load data ingestion completed successfully!")

if __name__ == "__main__":
    main()