# Data & Methodology Report

## 1. Data Sources & Acquisition

The project utilizes high-quality daily financial data sourced from Yahoo Finance via the `yfinance` API. The dataset spans from **1990-01-01** to the present, covering multiple market regimes (Dot-com bubble, 2008 Financial Crisis, COVID-19 crash).

| Ticker | Instrument | Purpose |
|--------|------------|---------|
| **SPY** | SPDR S&P 500 ETF Trust | Proxy for the S&P 500 index. Used to calculate returns and Realized Volatility (RV). |
| **^VIX** | CBOE Volatility Index | Market's expectation of 30-day forward volatility. Used as a feature (IV) and benchmark. |

---

## 2. Feature Engineering & Target Definition

### 2.1 Log Returns
We compute daily log returns for SPY to ensure additivity and statistical tractability:
$$ r_t = \ln(P_t) - \ln(P_{t-1}) $$

**Important**: Log returns are computed from **Adjusted Close** prices (accounting for dividends and splits).

### 2.2 Realized Volatility (Target)

The target variable is the **h-day Forward Realized Volatility**, where $h \in \{2, 5, 10, 30\}$ (configurable horizons). Great care was taken to ensure **no lookahead bias**.

*   **Definition**: The annualized sample standard deviation of returns over the *next* $h$ trading days.
*   **Formula**:
    $$ RV_{t,h} = \sqrt{252} \times \text{std}(r_{t+1}, r_{t+2}, \dots, r_{t+h}) $$
    where $\text{std}(\cdot)$ is the **sample standard deviation** (with Bessel's correction, i.e., dividing by $n-1$):
    $$ \text{std}(r_{t+1}, \dots, r_{t+h}) = \sqrt{\frac{1}{h-1} \sum_{i=1}^{h} (r_{t+i} - \bar{r})^2}, \quad \bar{r} = \frac{1}{h}\sum_{i=1}^{h} r_{t+i} $$

*   **Implementation** (from `features.py`):
    ```python
    def compute_forward_realized_volatility(returns: pd.Series, horizon: int, ann_factor: int = 252) -> pd.Series:
        # Shift returns by -1 so rolling window captures t+1 to t+h
        future_returns = returns.shift(-1)
        rolling_std = future_returns.rolling(window=horizon, min_periods=horizon).std()
        # Shift back by h-1 so that at index t, we have std of returns t+1 to t+h
        rv = rolling_std.shift(-(horizon - 1)) * np.sqrt(ann_factor)
        rv.name = f"RV_fwd_{horizon}"
        return rv
    ```

#### 2.2.1 Proof of No Lookahead Bias

Let's trace through exactly what data is used at each step:

1. **`shift(-1)` on returns**: Creates `future_returns[t] = r[t+1]`
2. **`.rolling(h).std()`**: At index `t`, computes std of `future_returns[t-h+1 : t+1]`  
   = std of `r[t-h+2], r[t-h+3], ..., r[t+1]` (h values from original returns)
3. **`shift(-(h-1))`**: At index `t`, we now have the value that was at index `t+h-1`  
   = std of `r[t+h-1-h+2], ..., r[t+h-1+1]` = std of `r[t+1], ..., r[t+h]` ✓

**Verification**: At time $t$, the target `RV_fwd_h[t]` uses **only** returns $r_{t+1}, r_{t+2}, \ldots, r_{t+h}$ — all **strictly future** returns. No lookahead bias exists.

**Note**: Pandas `.std()` uses `ddof=1` by default (Bessel's correction), which is appropriate for sample standard deviation.

### 2.3 Backward Realized Volatility (Feature)

For LSTM models, we also compute **backward-looking realized volatility** as a feature:
$$ RV^{back}_{t,h} = \sqrt{252} \times \text{std}(r_{t-h+1}, r_{t-h+2}, \dots, r_{t}) $$

**Implementation** (from `features.py`):
```python
def compute_backward_realized_volatility(returns: pd.Series, horizon: int, ann_factor: int = 252) -> pd.Series:
    rv = returns.rolling(window=horizon, min_periods=horizon).std() * np.sqrt(ann_factor)
    rv.name = f"RV_back_{horizon}"
    return rv
```

This uses **only past data** (the trailing h returns up to and including time t) and serves as a baseline feature for predicting future volatility. The rolling window at index $t$ computes std of returns $r_{t-h+1}, \ldots, r_t$ — all past or current values. ✓

### 2.4 Implied Volatility (Feature)
The VIX index is quoted in annualized percentage points (e.g., 20.0). We convert this to a decimal format for consistency with RV:
$$ IV_t = \frac{VIX_t}{100} $$

**Note on VIX vs. Realized Volatility Horizon**:
- VIX represents the market's expectation of **30-day** forward volatility.
- Our target horizons include **2, 5, 10, and 30 days**.
- For the 30-day horizon, VIX is a direct comparable benchmark.
- For shorter horizons, VIX still captures the overall volatility regime and is useful as a feature.

---

## 3. Statistical Rigor & Diagnostics

Before modeling, we performed comprehensive statistical tests to validate assumptions for GARCH and LSTM models.

### 3.1 Stationarity (ADF Test)
Stationarity is a prerequisite for GARCH models and helps LSTM convergence.
*   **Test**: Augmented Dickey-Fuller (ADF)
*   **Null Hypothesis ($H_0$)**: The series has a unit root (non-stationary).

| Series | p-value | Result | Conclusion |
|--------|---------|--------|------------|
| **Log Returns** | $0.0000$ | Reject $H_0$ | **Stationary** (Fit for GARCH) |
| **Realized Vol (RV)** | $0.0000$ | Reject $H_0$ | **Stationary** (Fit for LSTM) |
| **Implied Vol (IV)** | $0.0000$ | Reject $H_0$ | **Stationary** (Fit for LSTM) |

### 3.2 Volatility Clustering (ARCH-LM Test)
We verified the presence of volatility clustering (periods of high variance followed by high variance), which justifies the use of GARCH models.
*   **Test**: ARCH-LM (Lagrange Multiplier)
*   **Result**: p-value $\approx 0.00$ (Strong rejection of "no ARCH effects").
*   **Visual Proof**: See `artifacts/plots/data_analysis/02_volatility_clustering.png`.

### 3.3 Normality & Fat Tails (Jarque-Bera Test)
Financial returns are often assumed to be Normal, but in reality, they exhibit "fat tails" (extreme events happen more often than expected).
*   **Test**: Jarque-Bera
*   **Result**: p-value $\approx 0.00$ (Reject Normality).
*   **Implication**: We cannot use a Normal distribution for the GARCH model.
*   **Solution**: We selected a **Skewed Student's t-distribution** (`dist='skewt'`) for the GARCH model to capture these tails.

### 3.4 Leverage Effect
We tested for the "Leverage Effect" (negative returns increasing volatility more than positive returns).
*   **Evidence**: Strong negative correlation between $r_t$ and $\sigma^2_{t+k}$.
*   **Model Selection**: Standard GARCH assumes symmetric response. We chose **EGARCH** (Exponential GARCH), which explicitly models this asymmetry.

---

## 4. Preprocessing Pipeline

### 4.1 Winsorization
To prevent extreme outliers (e.g., 2020 COVID crash) from destabilizing the LSTM gradients, we winsorize inputs at the **0.5th and 99.5th percentiles**.
*   *Note: We do NOT winsorize the target variable, as predicting extreme events is the goal.*
*   Winsorization bounds are computed **only on training data** to prevent data leakage.

### 4.2 Log Transformation
Volatility is strictly positive and log-normally distributed. For the LSTM, we log-transform both inputs and targets:
$$ y_{train} = \ln(RV_{target}) $$
$$ X_{train} = \ln(IV_{input}) $$
This stabilizes the variance and allows the model to learn relative errors rather than absolute ones.

### 4.3 Data Splitting (Temporal)
To respect the time-series nature of the data, we use strict temporal splitting (no shuffling):

| Split | Period | Purpose |
|-------|--------|---------|
| **Train** | 1993-01-01 to 2015-12-31 | Model parameter estimation |
| **Validation** | 2016-01-01 to 2019-12-31 | Hyperparameter tuning & Early Stopping |
| **Test** | 2020-01-01 to Present | Final Out-of-Sample Performance |

### 4.4 Target Leakage Prevention

Since the forward target $RV_{t,h}$ uses returns from $t+1$ to $t+h$, training samples near the end of the training period would have targets that use returns from the validation period. To prevent this **target leakage**, we introduce a gap:

**Implementation** (from `train_loop.py`):
```python
# Remove target leakage: drop training samples whose target horizon extends into validation
train_end_ts = pd.to_datetime(splits_cfg["train_end"])
# Find the cutoff: drop rows where target horizon would extend into validation
# 1.5x multiplier accounts for weekends/holidays when converting calendar to trading days
safe_train_end = train_end_ts - pd.Timedelta(days=int(h * 1.5))
train_df = train_df.loc[train_df.index <= safe_train_end]
```

**Verification**:
- Training data ends approximately $h \times 1.5$ calendar days before the validation start date.
- For h=10, this means ~15 calendar days gap, which safely covers 10 trading days.
- The actual training end date is logged during training for verification.

This ensures that **no future information leaks into the training process**.

---

## 5. Artifacts

The analysis generates the following visual proofs in `artifacts/plots/data_analysis/`:
1.  `01_market_overview.png`: History of SPY and VIX.
2.  `02_volatility_clustering.png`: Visual evidence of ARCH effects.
3.  `03_distribution_fat_tails.png`: Empirical distribution vs. Normal vs. Student-t.
4.  `04_autocorrelation_check.png`: ACF plots confirming mean independence but variance dependence.
5.  `05_leverage_effect.png`: Correlation bar chart showing the leverage effect.

---

## 6. Configuration

Key data settings in `configs/data.yaml`:
```yaml
data:
  s_and_p_proxy: "SPY"
  vix_ticker: "^VIX"
  start_date: "1990-01-01"
  horizons: [2, 5, 10, 30]  # Forecast horizons in trading days
  annualization_factor: 252
```

Key training settings in `configs/train.yaml`:
```yaml
horizons: [2, 5, 10, 30]  # Must match data.yaml

preprocessing:
  log_transform_cols: []  # Dynamically populated: RV_fwd_{h}, RV_back_{h}, IV
  winsorize_cols:
    RET_SPY: [0.005, 0.005]  # Clip top/bottom 0.5%
```

---

## 7. Data Pipeline Architecture

The data pipeline is cleanly separated into two stages:

### 7.1 Data Generation (`download_data.py` → `features.py`)
Run once to download and preprocess all data:
```bash
python scripts/download_data.py --config configs/data.yaml
```

This creates `data/processed/timeseries.pkl` containing:
- `RET_SPY`: Daily log returns
- `IV`: Implied volatility (VIX / 100)
- `RV_fwd_2`, `RV_fwd_5`, `RV_fwd_10`, `RV_fwd_30`: Forward realized volatility targets
- `RV_back_2`, `RV_back_5`, `RV_back_10`, `RV_back_30`: Backward realized volatility features

### 7.2 Training (`run_train.py` → `train_loop.py`)
Uses pre-computed data, applies preprocessing (log transform, winsorization), and trains models:
```bash
python scripts/run_train.py --train_cfg configs/train.yaml --model_cfg configs/model/lstm_vix.yaml
```

**Key Design Principle**: All RV columns are computed during data generation, not during training. This ensures:
1. Consistent target definitions across all models
2. No duplicate code
3. Clear separation of concerns
4. Data can be inspected before training

---

## 8. Known Considerations

### 8.1 VIX Horizon Mismatch (for short horizons)
VIX measures 30-day implied volatility. For horizons h < 30, this is not a perfect predictor, but it still provides valuable regime information. The 30-day horizon now allows direct VIX vs RV comparison.

### 8.2 LSTM Sequence Handling
The LSTM model uses a sequence of past observations to predict future volatility. The `TimeSeriesDataset` aligns predictions to the **last** timestamp in each sequence window:
```python
# TimeSeriesDataset.__getitem__:
return self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len - 1]
```
This means the prediction at index `t` uses features from `t - seq_len + 1` to `t`, and the target is `RV_fwd_h[t]` (volatility from `t+1` to `t+h`). ✓

### 8.3 Winsorization Applied Only to Features
Returns are winsorized (clipped at 0.5th/99.5th percentiles) to stabilize training, but **the target (RV_fwd_h) is NOT winsorized**. This is intentional — we want the model to learn to predict extreme volatility events.

---

## 9. Usage

```python
import pandas as pd

# Load processed data (all columns pre-computed)
df = pd.read_pickle('data/processed/timeseries.pkl')

# Available columns:
# - RET_SPY: Daily log returns
# - IV: Implied volatility (VIX / 100)
# - RV_fwd_2, RV_fwd_5, RV_fwd_10, RV_fwd_30: Forward RV targets
# - RV_back_2, RV_back_5, RV_back_10, RV_back_30: Backward RV features

print(df.columns.tolist())
# ['RET_SPY', 'IV', 'RV_fwd_2', 'RV_back_2', 'RV_fwd_5', 'RV_back_5', 
#  'RV_fwd_10', 'RV_back_10', 'RV_fwd_30', 'RV_back_30']
```
