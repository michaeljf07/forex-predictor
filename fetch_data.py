import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import yfinance as yf
from datetime import datetime, timedelta

class ForexDataCollector:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        

    def fetch_forex_pair(self, ticker):
        data = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
        return data
    

    def fetch_multiple_pairs(self):
        pairs = {
            'EUR_USD': 'EURUSD=X',
            'GBP_USD': 'GBPUSD=X',
            'USD_JPY': 'JPY=X',
            'USD_CHF': 'CHF=X',
            'AUD_USD': 'AUDUSD=X',
            'USD_CAD': 'CAD=X',
            'NZD_USD': 'NZDUSD=X'
        }
        
        forex_data = {}
        for name, ticker in pairs.items():
            print(f"Fetching {name}...")
            forex_data[name] = self.fetch_forex_pair(ticker)
        
        return forex_data
    

    def fetch_market_indices(self):
        indices = {
            'SP500': '^GSPC',      # S&P 500
            'FTSE': '^FTSE',       # FTSE 100
            'DAX': '^GDAXI',       # DAX
            'NIKKEI': '^N225',     # Nikkei
            'ASX': '^AXJO'         # ASX
        }
        
        index_data = {}
        for name, ticker in indices.items():
            print(f"Fetching {name}...")
            index_data[name] = self.fetch_forex_pair(ticker)
        
        return index_data
    
    
    def fetch_commodities(self):
        commodities = {
            'GOLD': 'GC=F',        # Gold futures
            'OIL': 'CL=F',         # Crude oil
            'SILVER': 'SI=F',      # Silver
            'COPPER': 'HG=F'       # Copper
        }
        
        commodity_data = {}
        for name, ticker in commodities.items():
            print(f"Fetching {name}...")
            commodity_data[name] = self.fetch_forex_pair(ticker)
        
        return commodity_data
    

    def fetch_bond_yields(self):
        """Fetch government bond yields (proxy for interest rates)"""
        bonds = {
            'US_10Y': '^TNX',      # US 10-Year Treasury
            'US_2Y': '^IRX',       # US 2-Year Treasury (closer to policy rate)
        }
        
        bond_data = {}
        for name, ticker in bonds.items():
            print(f"Fetching {name}...")
            bond_data[name] = self.fetch_forex_pair(ticker)
        
        return bond_data
    

    def fetch_volatility_index(self):
        """Fetch VIX (market fear index)"""
        print("Fetching VIX...")
        vix = self.fetch_forex_pair('^VIX')
        return vix


end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

collector = ForexDataCollector(start_date, end_date)

print("=== Fetching Forex Pairs ===")
forex_data = collector.fetch_multiple_pairs()

print("\n=== Fetching Market Indices ===")
indices = collector.fetch_market_indices()

print("\n=== Fetching Commodities ===")
commodities = collector.fetch_commodities()

print("\n=== Fetching Bond Yields ===")
bonds = collector.fetch_bond_yields()

print("\n=== Fetching VIX ===")
vix = collector.fetch_volatility_index()

print("\n✅ Data collection complete!")
print(f"EUR/USD shape: {forex_data['EUR_USD'].shape}")