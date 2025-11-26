# Data Documentation

## Overview

The project uses financial time series data to train volatility prediction models and backtest trading strategies.

## Data Sources

### Primary Data
- **SPY ETF**: S&P 500 ETF daily prices and returns
- **VIX Index**: CBOE Volatility Index (implied volatility)
- **Risk-free Rate**: Treasury rates for Sharpe ratio calculations

### Data Location
- Raw data: `data/raw/`
- Processed data: `data/processed/timeseries.pkl`

## Key Variables

| Variable | Description | Source |
|----------|-------------|--------|
| `RET_SPY` | Daily log returns of SPY | Calculated from prices |
| `RV30` | 30-day realized volatility | Rolling std of returns |
| `IV` | Implied volatility (VIX / 100) | CBOE VIX Index |

## Preprocessing Pipeline

### 1. Data Download
```bash
python scripts/download_data.py
```
Downloads SPY and VIX data from Yahoo Finance using `yfinance`.

### 2. Feature Engineering

#### Realized Volatility (RV)
For horizon `h`, we compute:
- **Forward RV**: `RV_fwd_h` - next `h` days realized volatility (prediction target)
- **Backward RV**: `RV_back_h` - past `h` days realized volatility (model input)

```python
# Forward RV (target)
RV_fwd_h = returns.rolling(h).std().shift(-h) * sqrt(252)

# Backward RV (feature)
RV_back_h = returns.rolling(h).std() * sqrt(252)
```

#### Log Transformation
All volatility values are log-transformed for model training:
```python
log_rv = np.log(rv + epsilon)  # epsilon = 1e-8 for numerical stability
```

This is critical because:
1. Volatility is always positive but can be near zero
2. Log-space predictions can be negative (which `softplus` would clip incorrectly)
3. Errors in log-space are relative, not absolute

### 3. Train/Val/Test Split

| Split | Period | Purpose |
|-------|--------|---------|
| Train | 1993-01-01 to 2010-12-31 | Model training |
| Val | 2011-01-01 to 2015-12-31 | Hyperparameter tuning, early stopping |
| Test | 2016-01-01 to present | Out-of-sample evaluation |

### 4. Stationarity Checks
We verify stationarity using Augmented Dickey-Fuller (ADF) tests:
- Log returns: Stationary ✓
- Log volatility: Stationary ✓
- VIX levels: Near-stationary (mean-reverting)

## Data Quality Notes

### Missing Values
- Weekend/holiday gaps are handled by pandas DatetimeIndex
- Early periods may have missing VIX data (pre-1990)

### Outliers
- Winsorization applied at 1st and 99th percentiles
- Major events (2008 crisis, COVID crash) retained for training

### Lookahead Bias Prevention
- All features computed using only past data
- Target (RV_fwd) is forward-looking by definition
- Careful alignment ensures no data leakage

## Configuration

Data configuration is in `configs/data.yaml`:

```yaml
data:
  ticker: SPY
  start_date: "1993-01-01"
  train_end: "2010-12-31"
  val_end: "2015-12-31"
  
preprocessing:
  log_transform: true
  winsorize: true
  winsorize_limits: [0.01, 0.99]
```

## Usage

```python
import pandas as pd

# Load processed data
df = pd.read_pickle('data/processed/timeseries.pkl')

# Access key columns
spy_returns = df['RET_SPY']
realized_vol = df['RV30']
implied_vol = df['IV']
```
