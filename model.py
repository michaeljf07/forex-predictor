import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

from fetch_data import collector, forex_data, indices, commodities, bonds, vix
from features import FeatureEngineer


class ForexModelTrainer:
    def __init__(self, target_pair='EUR_USD', forecast_days=1):
        self.target_pair = target_pair
        self.forecast_days = forecast_days
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_features(self, forex_data, indices, commodities, bonds, vix):
        print(f"\n=== Preparing features for {self.target_pair} ===")
        
        target_data = forex_data[self.target_pair]
        print(f"Target data shape: {target_data.shape}")
        print(f"Date range: {target_data.index[0]} to {target_data.index[-1]}")
        
        engineer = FeatureEngineer(target_data)
        
        print("Adding technical indicators...")
        engineer.add_technical_indicators()
        
        print("Adding time features...")
        engineer.add_time_features()
        
        print("Adding market indices, commodities, bonds, and VIX...")
        engineer.add_market_data(indices, commodities, bonds, vix)
        
        print("Adding cross-pair features...")
        other_pairs = {k: v for k, v in forex_data.items() if k != self.target_pair}
        engineer.add_cross_pair_features(other_pairs)
        
        print(f"Creating target (forecast_days={self.forecast_days})...")
        engineer.create_target(forecast_days=self.forecast_days)
        
        df = engineer.get_data()
        
        # drop initial rows with too many NaN values
        print(f"\nBefore dropping NaN rows: {len(df)}")
        
        # drop rows where more than 50% of features are NaN
        threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=threshold)
        
        print(f"After dropping sparse rows: {len(df)}")
        
        if len(df) < 100:
            print(f"WARNING: Only {len(df)} rows remaining. This may be insufficient for training.")
        
        return df
    
    
    def prepare_train_test_data(self, df):
        print("\n=== Preparing train/test split ===")
        print(f"Initial DataFrame shape: {df.shape}")
        
        # define features to exclude
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
                       'Target', 'Target_Return', 'Target_Direction']
        
        # select feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"Number of features: {len(feature_cols)}")
        
        # remove rows with NaN in target first
        df_clean = df.dropna(subset=['Target_Return'])
        print(f"After removing NaN targets: {len(df_clean)} rows")
        
        if len(df_clean) == 0:
            raise ValueError("No valid data after removing NaN targets!")
        
        X = df_clean[feature_cols]
        y = df_clean['Target_Return']
        
        # identify columns with too many NaN values (>80% missing)
        nan_counts = X.isna().sum()
        nan_pct = nan_counts / len(X) * 100
        high_nan_cols = nan_pct[nan_pct > 80].index.tolist()
        
        if len(high_nan_cols) > 0:
            print(f"\nDropping {len(high_nan_cols)} features with >80% missing data:")

            for col in high_nan_cols:
                print(f"  - {col}: {nan_pct[col]:.1f}% missing")

            X = X.drop(columns=high_nan_cols)
            feature_cols = [col for col in feature_cols if col not in high_nan_cols]
        
        # fill remaining NaN values with forward fill then backward fill
        print(f"\nFilling remaining NaN values...")
        X = X.ffill().bfill()
        
        # if still NaN values remain, fill with 0
        remaining_nans = X.isna().sum().sum()
        if remaining_nans > 0:
            print(f"Warning: {remaining_nans} NaN values remaining, filling with 0")
            X = X.fillna(0)
        
        print(f"\nTotal valid samples: {len(X)}")
        print(f"Features after cleanup: {len(feature_cols)}")
        
        if len(X) == 0:
            raise ValueError("No valid data remaining after cleanup!")
        
        # Time series split: use last 20% for testing
        split_idx = int(len(X) * 0.8)
        
        if split_idx == 0:
            raise ValueError(f"Not enough data for train/test split. Only {len(X)} samples available.")
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Train samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("Train or test set is empty after split!")
        
        self.feature_names = feature_cols
        
        return X_train, X_test, y_train, y_test
    

    def train_models(self, X_train, y_train):
        print("\n=== Training models ===")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # define model
        model_configs = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            ),
            'Ridge Regression': Ridge(
                alpha=1.0,
                random_state=42
            )
        }
        
        # train each model
        for name, model in model_configs.items():
            print(f"\nTraining {name}...")
            
            model.fit(X_train_scaled, y_train)
            self.models[name] = model

            print(f"{name} trained successfully!")
        
        return self.models
    
    
    def evaluate_models(self, X_test, y_test):
        print("\n=== Model Evaluation ===")
        
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'predictions': y_pred
            }
            
            print(f"\n{name}:")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  R²: {r2:.4f}")
        
        return results
    
    
    def get_feature_importance(self, model_name='Random Forest', top_n=20):
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return feature_importance_df.head(top_n)
        
        else:
            print(f"{model_name} does not have feature_importances_ attribute")
            return None
    

    def save_models(self, filepath_prefix='models/forex_model'):
        print(f"\n=== Saving models ===")
        
        for name, model in self.models.items():
            filename = f"{filepath_prefix}_{name.replace(' ', '_').lower()}.pkl"
            joblib.dump(model, filename)
            print(f"Saved {name} to {filename}")
        
        scaler_file = f"{filepath_prefix}_scaler.pkl"
        joblib.dump(self.scaler, scaler_file)
        print(f"Saved scaler to {scaler_file}")
        
        features_file = f"{filepath_prefix}_features.pkl"
        joblib.dump(self.feature_names, features_file)
        print(f"Saved feature names to {features_file}")
    

    def cross_validate(self, X, y, n_splits=5):
        print("\n=== Time Series Cross-Validation ===")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_results = {name: [] for name in self.models.keys()}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            print(f"\nFold {fold}/{n_splits}")
            
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)
            
            # evaluate each model
            for name, model in self.models.items():
                model.fit(X_train_scaled, y_train_fold)
                y_pred = model.predict(X_val_scaled)
                rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                cv_results[name].append(rmse)
        
        print("\n=== Cross-Validation Results (Average RMSE) ===")
        for name, scores in cv_results.items():
            print(f"{name}: {np.mean(scores):.6f} (+/- {np.std(scores):.6f})")
        
        return cv_results


if __name__ == "__main__":
    import os
    
    os.makedirs('models', exist_ok=True)
    
    print("="*60)
    print("FOREX PRICE PREDICTION MODEL TRAINING")
    print("="*60)
    
    trainer = ForexModelTrainer(target_pair='EUR_USD', forecast_days=1)
    
    # Prepare features
    df = trainer.prepare_features(forex_data, indices, commodities, bonds, vix)
    
    print(f"\nDataFrame shape after feature engineering: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # prepare test data
    X_train, X_test, y_train, y_test = trainer.prepare_train_test_data(df)
    
    # train models
    models = trainer.train_models(X_train, y_train)
    
    results = trainer.evaluate_models(X_test, y_test)
    
    print("\n=== Top 20 Most Important Features ===")
    importance_df = trainer.get_feature_importance('Random Forest', top_n=20)
    if importance_df is not None:
        print(importance_df.to_string(index=False))
    
    # perform cross-validation
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    cv_results = trainer.cross_validate(X_full, y_full, n_splits=5)
    
    trainer.save_models()
    
    # save the results for visualization
    import pickle
    with open('models/training_results.pkl', 'wb') as f:
        pickle.dump({
            'trainer': trainer,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'results': results
        }, f)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)