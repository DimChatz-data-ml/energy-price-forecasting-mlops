"""
Fetch Generation by Fuel Type (Long Format) from ENTSO-E
Saves to: raw_generation_long (Postgres)
Format: (country_code, period_start, fuel_type, gen_mw)
Features: 7-day chunks, MultiIndex handling, fuel aggregation, transactional upsert
"""
import os
import logging
import time
import pandas as pd
from datetime import datetime
from entsoe import EntsoePandasClient
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("ENTSOE_API_KEY")
DB_URL = os.getenv("DATABASE_URL")

# Target countries: DE_LU last for fault isolation
COUNTRIES = ["GR", "FR", "ES", "PL", "DE_LU"]
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def get_engine():
    """Create SQLAlchemy engine with connection health check."""
    return create_engine(DB_URL, pool_pre_ping=True)


def ensure_long_table(engine):
    """Create raw_generation_long table with composite PRIMARY KEY."""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw_generation_long (
                country_code VARCHAR(10) NOT NULL,
                period_start TIMESTAMPTZ NOT NULL,
                fuel_type VARCHAR(100) NOT NULL,
                gen_mw DECIMAL(18,2),
                ingestion_timestamp TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (country_code, period_start, fuel_type)
            );
        """))
        conn.commit()
        logging.info("✅ Table raw_generation_long ensured with PK constraint.")


def fetch_gen_chunk(client, country, start, end):
    """
    Fetch generation data for one country, one time window.
    Returns DataFrame in long format: (country_code, period_start, fuel_type, gen_mw)
    """
    try:
        df = client.query_generation(country, start=start, end=end)
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Handle MultiIndex columns (common in entsoe-py responses)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index().rename(columns={'index': 'period_start'})
        
        # Map ENTSO-E fuel names to standardized types
        fuel_mapping = {
            'Fossil Hard coal': 'Coal',
            'Fossil Brown coal/Lignite': 'Coal',
            'Fossil Gas': 'Gas',
            'Nuclear': 'Nuclear',
            'Wind Onshore': 'Wind',
            'Wind Offshore': 'Wind',
            'Solar': 'Solar',
            'Geothermal': 'Geothermal',
            'Hydro Run-of-river and poundage': 'Hydro',
            'Hydro Water Reservoir': 'Hydro',
            'Hydro Pumped Storage': 'Hydro',
            'Biomass': 'Biomass',
            'Other renewable': 'Other',
            'Other': 'Other'
        }
        
        # Keep only columns that exist AND are mappable
        available_cols = [c for c in df.columns if c in fuel_mapping.keys()]
        if not available_cols:
            return pd.DataFrame()
        
        # Melt to long format
        df_long = df.melt(
            id_vars=['period_start'],
            value_vars=available_cols,
            var_name='raw_fuel',
            value_name='gen_mw'
        )
        
        # Map to standardized fuel types and drop unmapped rows
        df_long['fuel_type'] = df_long['raw_fuel'].map(fuel_mapping)
        df_long = df_long.dropna(subset=['fuel_type', 'gen_mw'])
        
        # Add metadata
        df_long['country_code'] = country
        df_long['period_start'] = pd.to_datetime(df_long['period_start']).dt.tz_convert('UTC')
        
        # Aggregate duplicate (country, period, fuel) combinations
        df_long = df_long.groupby(['country_code', 'period_start', 'fuel_type'], as_index=False)['gen_mw'].sum()
        
        return df_long[['country_code', 'period_start', 'fuel_type', 'gen_mw']]
        
    except Exception as e:
        logging.warning(f"⚠️ API error | {country} | {start.strftime('%Y-%m-%d')}: {str(e)[:60]}")
        return pd.DataFrame()


def save_chunk_transactional(df, engine):
    """Save DataFrame using temp table + UPSERT inside a transaction."""
    if df.empty:
        return
    
    with engine.begin() as conn:
        # Create temp table (auto-dropped on commit)
        conn.execute(text("""
            CREATE TEMP TABLE t_gen (
                country_code VARCHAR(10),
                period_start TIMESTAMPTZ,
                fuel_type VARCHAR(100),
                gen_mw DECIMAL(18,2)
            ) ON COMMIT DROP
        """))
        
        # Load data into temp table
        df.to_sql('t_gen', conn, if_exists='append', index=False, method='multi')
        
        # Upsert: insert new, ignore duplicates
       # Upsert: insert new, ignore duplicates
        conn.execute(text("""
            INSERT INTO public.raw_generation_long (country_code, period_start, fuel_type, gen_mw)
            SELECT country_code, period_start, fuel_type, gen_mw FROM t_gen
            ON CONFLICT (country_code, period_start, fuel_type) DO NOTHING
        """))

def main():
    # Validate environment
    if not API_KEY or not DB_URL:
        raise ValueError("❌ Missing ENTSOE_API_KEY or DATABASE_URL in .env")
    
    client = EntsoePandasClient(api_key=API_KEY)
    engine = get_engine()
    
    # Ensure table exists
    ensure_long_table(engine)
    
    for country in COUNTRIES:
        logging.info(f"🚀 Starting generation fetch for {country}...")
        
        current_start = pd.Timestamp(START_DATE, tz='UTC')
        end_limit = pd.Timestamp(END_DATE, tz='UTC')
        
        rows_saved = 0
        
        while current_start < end_limit:
            # 7-day chunks for stability
            current_end = min(current_start + pd.Timedelta(days=7), end_limit)
            
            chunk = fetch_gen_chunk(client, country, current_start, current_end)
            
            if not chunk.empty:
                save_chunk_transactional(chunk, engine)
                rows_saved += len(chunk)
                logging.info(f"📥 {country}: Saved {len(chunk)} rows for week starting {current_start.strftime('%Y-%m-%d')}")
            
            # Rate limiting: be polite to the API
            time.sleep(1.0)
            current_start = current_end
        
        logging.info(f"✅ {country}: Total {rows_saved} rows saved.")
    
    logging.info("🏁 Generation ingestion completed for all countries!")


if __name__ == "__main__":
    main()