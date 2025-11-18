import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, target_pair_data):
        # flatten multi-level columns if they exist
        if isinstance(target_pair_data.columns, pd.MultiIndex):
            target_pair_data.columns = target_pair_data.columns.get_level_values(0)
        
        self.data = target_pair_data.copy()
        

    def add_technical_indicators(self):
        df = self.data
        
        # Ensure we're working with a single Close column
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must have a 'Close' column")
        
        close = df['Close'].squeeze()  # Ensure it's a Series
        
        # Price-based features
        df['Returns'] = close.pct_change()
        df['Log_Returns'] = np.log(close / close.shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            df[f'MA_{window}'] = close.rolling(window=window).mean()
            df[f'MA_{window}_diff'] = close - df[f'MA_{window}']
        
        # Exponential moving averages
        for span in [12, 26]:
            df[f'EMA_{span}'] = close.ewm(span=span, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (close - df['BB_Lower']) / df['BB_Width']
        
        # volatility
        for window in [5, 10, 20, 30]:
            df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()
        
        # momentum
        for period in [5, 10, 20]:
            df[f'Momentum_{period}'] = close - close.shift(period)
            df[f'ROC_{period}'] = close.pct_change(periods=period)
        
        # RSI (Relative Strength Index)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        if 'Low' in df.columns and 'High' in df.columns:
            low = df['Low'].squeeze()
            high = df['High'].squeeze()
            low_14 = low.rolling(window=14).min()
            high_14 = high.rolling(window=14).max()
            df['Stochastic'] = 100 * (close - low_14) / (high_14 - low_14)
            
            # average True Range (ATR)
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['ATR'] = true_range.rolling(14).mean()
        
        # volume features
        if 'Volume' in df.columns:
            volume = df['Volume'].squeeze()
            df['Volume_MA_20'] = volume.rolling(window=20).mean()
            df['Volume_Ratio'] = volume / df['Volume_MA_20']
        
        self.data = df
        return self
    

    def add_time_features(self):
        df = self.data
        
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Day_of_Month'] = df.index.day
        df['Week_of_Year'] = df.index.isocalendar().week
        
        # cyclical encoding for day of week and month
        df['Day_Sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
        df['Day_Cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        self.data = df
        return self
    

    def add_market_data(self, indices_dict, commodities_dict, bonds_dict, vix_data):
        df = self.data
        
        # extract Close column safely
        def get_close(data):
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            return data['Close'].squeeze()
        
        # add stock indices
        for name, index_data in indices_dict.items():
            close = get_close(index_data)
            df[f'{name}_Close'] = close
            df[f'{name}_Returns'] = close.pct_change()
            df[f'{name}_Volatility'] = df[f'{name}_Returns'].rolling(20).std()
        
        # add commodities
        for name, commodity_data in commodities_dict.items():
            close = get_close(commodity_data)
            df[f'{name}_Close'] = close
            df[f'{name}_Returns'] = close.pct_change()
        
        # add bond yields
        for name, bond_data in bonds_dict.items():
            close = get_close(bond_data)
            df[f'{name}_Yield'] = close
            df[f'{name}_Change'] = close.diff()
        
        # add yield spread
        if 'US_10Y_Yield' in df.columns and 'US_2Y_Yield' in df.columns:
            df['Yield_Spread'] = df['US_10Y_Yield'] - df['US_2Y_Yield']
        
        # add VIX
        vix_close = get_close(vix_data)
        df['VIX'] = vix_close
        df['VIX_Change'] = vix_close.pct_change()
        
        # Forward fill missing values (market data on different schedules)
        df = df.ffill().bfill()
        
        self.data = df
        return self
    
    def add_cross_pair_features(self, other_pairs_dict):
        df = self.data
        
        def get_close(data):
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            return data['Close'].squeeze()
        
        for pair_name, pair_data in other_pairs_dict.items():
            close = get_close(pair_data)
            df[f'{pair_name}_Close'] = close
            df[f'{pair_name}_Returns'] = close.pct_change()
            
            # calculate rolling correlation
            if 'Returns' in df.columns:
                corr = df['Returns'].rolling(30).corr(df[f'{pair_name}_Returns'])
                df[f'{pair_name}_Correlation'] = corr
        
        df = df.ffill().bfill()
        self.data = df
        return self
    

    def create_target(self, forecast_days=1):
        df = self.data
        
        close = df['Close'].squeeze()
        
        df['Target'] = close.shift(-forecast_days)
        df['Target_Return'] = (df['Target'] / close) - 1
        df['Target_Direction'] = (df['Target'] > close).astype(int)
        
        self.data = df
        return self
    

    def get_data(self):
        return self.data
    
    
