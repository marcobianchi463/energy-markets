import pandas as pd
from entsoe import EntsoePandasClient

def load_day_ahead_prices(api_key: str, country: str = "IT_NORD", start: str = "2022-01-01", end: str = "2024-01-01") -> pd.Series:
    """
    Downloads day-ahead prices from the ENTSO-E electric market.
    
    Args:
    - api_key: ENTSO-E API key
    - country: country code (IT, DE, ES, FR, ...)
    - start: starting date (YYYY-MM-DD)
    - end: ending date (YYYY-MM-DD)
    
    Returns:
    pandas Series with hourly prices in EUR/MWh
    """
    client = EntsoePandasClient(api_key=api_key)

    start_ts = pd.Timestamp(start, tz="Europe/Rome")
    end_ts = pd.Timestamp(end, tz="Europe/Rome")

    prices = client.query_day_ahead_prices(country, start=start_ts, end=end_ts)
    return prices

def load_generation(api_key: str, country: str = "IT_NORD", start: str = "2022-01-01", end: str = "2024-01-01") -> pd.DataFrame:
    """
    Downloads per source electricity production (solar, wind, gas, nuclear, etc.)
    """
    client = EntsoePandasClient(api_key=api_key)

    start_ts = pd.Timestamp(start, tz="Europe/Rome")
    end_ts = pd.Timestamp(end, tz="Europe/Rome")

    generation = client.query_generation(country, start=start_ts, end=end_ts, psr_type=None)

    return generation

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()

    API_KEY = os.getenv("ENTSOE_API_KEY")

    prices = load_day_ahead_prices(API_KEY)
    # generation = load_generation(API_KEY)

    prices.to_csv("data/raw/prices_IT_2022_2024.csv")
    # generation.to_csv("data/raw/generation_IT_2022_2024.csv")

    print(f"Prices: {len(prices)} observations")
    # print(f"Production: {generation.shape}")