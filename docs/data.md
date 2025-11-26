# Data & Methodology Report

## 1. Data Sources & Acquisition

The project utilizes high-quality daily financial data sourced from Yahoo Finance via the `yfinance` API. The dataset spans from **1993-01-01** to the present, covering multiple market regimes (Dot-com bubble, 2008 Financial Crisis, COVID-19 crash).

| Ticker | Instrument | Purpose |
|--------|------------|---------|
| **SPY** | SPDR S&P 500 ETF Trust | Proxy for the S&P 500 index. Used to calculate returns and Realized Volatility (RV). |
| **^VIX** | CBOE Volatility Index | Market's expectation of 30-day forward volatility. Used as a feature (IV) and benchmark. |

---

## 2. Feature Engineering & Target Definition

### 2.1 Log Returns
We compute daily log returns for SPY to ensure additivity and statistical tractability:
$$ r_t = \ln(P_t) - \ln(P_{t-1}) $$

### 2.2 Realized Volatility (Target)
The target variable is the **30-day Forward Realized Volatility**. Great care was taken to ensure **no lookahead bias**.

*   **Definition**: The annualized standard deviation of returns over the *next* 30 trading days.
*   **Formula**:
    $$ RV_{t, 30} = \sqrt{252} \times \text{std}(r_{t+1}, r_{t+2}, \dots, r_{t+30}) $$
*   **Implementation**:
    ```python
    # Calculate rolling std (backward looking by default)
    rolling_std = r.rolling(window=30).std()
    # Shift backward by window size to align future volatility to current time t
    target = rolling_std.shift(-30)
    ```
    *At time $t$, the target value depends strictly on returns from $t+1$ to $t+30$.*

### 2.3 Implied Volatility (Feature)
The VIX index is quoted in annualized percentage points (e.g., 20.0). We convert this to a decimal format for consistency with RV:
$$ IV_t = \frac{VIX_t}{100} $$

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
| **Realized Vol (RV30)** | $0.0000$ | Reject $H_0$ | **Stationary** (Fit for LSTM) |
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
To prevent extreme outliers (e.g., 2020 COVID crash) from destabilizing the LSTM gradients, we winsorize inputs at the **1st and 99th percentiles**.
*   *Note: We do NOT winsorize the target variable, as predicting extreme events is the goal.*

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

---

## 5. Artifacts

The analysis generated the following visual proofs in `artifacts/plots/data_analysis/`:
1.  `01_market_overview.png`: History of SPY and VIX.
2.  `02_volatility_clustering.png`: Visual evidence of ARCH effects.
3.  `03_distribution_fat_tails.png`: Empirical distribution vs. Normal vs. Student-t.
4.  `04_autocorrelation_check.png`: ACF plots confirming mean independence but variance dependence.
5.  `05_leverage_effect.png`: Correlation bar chart showing the leverage effect.
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
