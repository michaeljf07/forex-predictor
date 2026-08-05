# Forex Price Prediction with Machine Learning

A machine learning pipeline for predicting forex price movements using technical indicators and market data. Built to explore quantitative finance and learn ML techniques.

## Overview

This project trains multiple ML models to predict next-day returns for major currency pairs using 90+ engineered features from technical analysis, market indices, commodities, and bond yields.

**Models**: Random Forest, Gradient Boosting, Ridge Regression  
**Data**: 5 years of daily forex, equity indices, commodities, bonds, VIX  
**Features**: Technical indicators (MACD, RSI, Bollinger Bands), time features, cross-market correlations

## Quick Start

```bash
# Install dependencies
pip install pandas numpy scikit-learn yfinance matplotlib seaborn scipy joblib

# Run pipeline
python fetch_data.py          # Collect data
python model.py               # Train models
python visualize_results.py   # Generate plots
```

## Project Structure

```
├── fetch_data.py         # Data collection from Yahoo Finance
├── features.py           # Feature engineering (90+ indicators)
├── model.py             # Model training & evaluation
├── visualize_results.py # Performance visualization
├── models/              # Saved models & results
└── visualizations/      # Generated plots
```

## Features

### Data Collection

-   7 forex pairs (EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD)
-   5 market indices (S&P 500, FTSE, DAX, Nikkei, ASX)
-   4 commodities (Gold, Oil, Silver, Copper)
-   Bond yields (US 10Y, 2Y) and VIX

### Feature Engineering

-   **Technical**: MA, EMA, MACD, RSI, Bollinger Bands, Stochastic, ATR
-   **Price**: Returns, log returns, momentum, ROC
-   **Volatility**: Multi-timeframe standard deviations
-   **Time**: Cyclical encoding (day/month)
-   **Cross-Market**: Currency pair correlations, yield spreads

### Visualization Suite

-   Model comparison (RMSE, MAE, R²)
-   Prediction vs actual plots
-   Residual analysis with Q-Q plots
-   Feature importance rankings
-   Cumulative returns (ML strategy vs buy-and-hold)
-   Price forecast charts

**Top Features**: Recent returns, MA differences, RSI, cross-pair correlations

## What I Learned

**Economics & Finance**: Technical analysis, macro indicators, market relationships, efficient market hypothesis

**Machine Learning**: Time series features, train/test splitting for sequential data, ensemble methods, cross-validation, model evaluation

**Engineering**: Modular Python design, data pipelines, model persistence, reproducible experiments

## Example Images

![Random Forest Cumulative Returns](https://github.com/michaeljf07/forex-predictor/blob/main/visualizations/Random_Forest_cumulative_returns.png)
![Random Forest Feature Importance](https://github.com/michaeljf07/forex-predictor/blob/main/visualizations/Random_Forest_feature_importance.png)
![Random Forest Predictions](https://github.com/michaeljf07/forex-predictor/blob/main/visualizations/Random_Forest_predictions.png)
![Random Forest Price Predictions](https://github.com/michaeljf07/forex-predictor/blob/main/visualizations/Random_Forest_price_prediction.png)
![Random Forest Residuals](https://github.com/michaeljf07/forex-predictor/blob/main/visualizations/Random_Forest_residuals.png)
![Random Distribution](https://github.com/michaeljf07/forex-predictor/blob/main/visualizations/error_distribution.png)
![Random Comparison](https://github.com/michaeljf07/forex-predictor/blob/main/visualizations/model_comparison.png)
![Summary Report](https://github.com/michaeljf07/forex-predictor/blob/main/visualizations/summary_report.png)

## Disclaimer

⚠️ **Educational purposes only.** Do not use for actual trading. Forex trading involves substantial risk. Past performance ≠ future results. Consult a financial advisor before investing.
