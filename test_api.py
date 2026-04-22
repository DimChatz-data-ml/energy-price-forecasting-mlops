import os
from entsoe import EntsoePandasClient
import pandas as pd
from dotenv import load_dotenv

# Φόρτωση του API Key από το .env
load_dotenv()
api_key = os.getenv('ENTSOE_API_KEY')

# Δημιουργία του Client
client = EntsoePandasClient(api_key=api_key)

# Ορίζουμε το χρονικό διάστημα (π.χ. τις τελευταίες 24 ώρες)
end = pd.Timestamp.now(tz='UTC')
start = end - pd.Timedelta(days=1)

# Ορίζουμε τη χώρα (GR για Ελλάδα)
country_code = 'GR'

try:
    print(f"📡 Προσπάθεια σύνδεσης με ENTSO-E για την χώρα: {country_code}...")
    
    # Ζητάμε τις τιμές του ρεύματος (Day-Ahead Prices)
    prices = client.query_day_ahead_prices(country_code, start=start, end=end)
    
    print("✅ Επιτυχία! Λάβαμε δεδομένα:")
    print(prices.head())
    
except Exception as e:
    print(f"❌ Κάτι πήγε στραβά: {e}")