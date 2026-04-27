import os
import logging
import time
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from entsoe import EntsoePandasClient
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ENTSOE_API_KEY")
DB_URL = os.getenv("DATABASE_URL")

COUNTRIES = ["GR", "DE_LU", "FR", "ES", "PL"]
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)

def fetch_monthly_chunk(client, country_code, start_date, end_date):
    try:
        df = client.query_day_ahead_prices(country_code, start=start_date, end=end_date)
        if df is not None and not df.empty:
            df = df.reset_index()
            df.columns = ['period_start', 'price_eur_mwh']
            df['country_code'] = country_code
            df['period_start'] = pd.to_datetime(df['period_start']).dt.tz_convert('UTC')
            return df
    except Exception as e:
        logging.warning(f"⚠️ Αποτυχία λήψης {country_code} για {start_date.strftime('%Y-%m')}: {e}")
    return pd.DataFrame()

def validate_data(df, country_code):
    original_len = len(df)
    df = df.drop_duplicates(subset=['country_code', 'period_start'])
    df = df[df['price_eur_mwh'].between(-500, 5000)]
    
    if len(df) != original_len:
        logging.info(f"🧹 Καθαρίστηκαν {original_len - len(df)} εγγραφές για {country_code}")
    return df

def main():
    if not API_KEY or not DB_URL:
        raise ValueError("❌ Λείπει το API key ή το Database URL στο αρχείο .env")
    
    client = EntsoePandasClient(api_key=API_KEY)
    engine = get_engine()
    
    for country in COUNTRIES:
        logging.info(f"🚀 Ξεκινάει η λήψη δεδομένων για: {country}...")
        start = pd.Timestamp(START_DATE, tz='UTC')
        end_limit = pd.Timestamp(END_DATE, tz='UTC')
        
        country_data = []
        while start < end_limit:
            chunk_end = min(start + relativedelta(months=1), end_limit)
            
            chunk = fetch_monthly_chunk(client, country, start, chunk_end)
            
            if not chunk.empty:
                chunk = validate_data(chunk, country)
                country_data.append(chunk)
            
            time.sleep(0.5) 
            start = chunk_end

        if country_data:
            country_df = pd.concat(country_data, ignore_index=True)
            country_df = country_df.drop_duplicates(subset=['country_code', 'period_start'])
            
            logging.info(f"💾 Αποθήκευση {len(country_df)} γραμμών για {country} στη βάση...")
            
            try:
                with engine.begin() as conn:
                    conn.execute(text("CREATE TEMP TABLE temp_prices AS SELECT * FROM raw_day_ahead_prices WITH NO DATA"))
                    country_df.to_sql('temp_prices', conn, if_exists='append', index=False)
                    
                    upsert_query = text("""
                        INSERT INTO raw_day_ahead_prices (country_code, period_start, price_eur_mwh)
                        SELECT country_code, period_start, price_eur_mwh FROM temp_prices
                        ON CONFLICT (country_code, period_start) 
                        DO UPDATE SET price_eur_mwh = EXCLUDED.price_eur_mwh;
                    """)
                    conn.execute(upsert_query)
                    conn.execute(text("DROP TABLE temp_prices"))
                    
                logging.info(f"✅ Τα δεδομένα της χώρας {country} αποθηκεύτηκαν επιτυχώς!")
            except Exception as e:
                logging.error(f"❌ Σφάλμα κατά την αποθήκευση της χώρας {country}: {e}")
        else:
            logging.warning(f"⏭️ Δεν βρέθηκαν δεδομένα για τη χώρα {country}.")

    logging.info("🏁 Η διαδικασία ολοκληρώθηκε για όλες τις χώρες!")

if __name__ == "__main__":
    main()