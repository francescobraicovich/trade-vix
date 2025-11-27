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

The target variable is the **h-day Forward Realized Volatility**, where $h \in \{2, 5, 10\}$ (configurable horizons). Great care was taken to ensure **no lookahead bias**.

*   **Definition**: The annualized sample standard deviation of returns over the *next* $h$ trading days.
*   **Formula**:
    $$ RV_{t,h} = \sqrt{252} \times \text{std}(r_{t+1}, r_{t+2}, \dots, r_{t+h}) $$
    where $\text{std}(\cdot)$ is the **sample standard deviation** (with Bessel's correction, i.e., dividing by $n-1$):
    $$ \text{std}(r_{t+1}, \dots, r_{t+h}) = \sqrt{\frac{1}{h-1} \sum_{i=1}^{h} (r_{t+i} - \bar{r})^2}, \quad \bar{r} = \frac{1}{h}\sum_{i=1}^{h} r_{t+i} $$
*   **Implementation**:
    ```python
    # Shift returns by -1 so rolling window captures t+1 to t+h
    future_returns = df[ret_col].shift(-1)
    rolling_std_fwd = future_returns.rolling(window=h, min_periods=h).std()
    # Shift back by h-1 so that at index t, we have std of returns t+1 to t+h
    df[f"RV_fwd_{h}"] = rolling_std_fwd.shift(-(h - 1)) * np.sqrt(252)
    ```
    *At time $t$, the target value depends strictly on returns from $t+1$ to $t+h$.*

### 2.3 Backward Realized Volatility (Feature)

For LSTM models, we also compute **backward-looking realized volatility** as a feature:
$$ RV^{back}_{t,h} = \sqrt{252} \times \text{std}(r_{t-h+1}, r_{t-h+2}, \dots, r_{t}) $$

This uses **only past data** and serves as a baseline feature for predicting future volatility.

### 2.4 Implied Volatility (Feature)
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

*   Training samples are excluded if their target horizon extends into the validation period.
*   Specifically, training data ends approximately $h \times 1.5$ calendar days before the validation start date (to account for weekends/holidays).

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

Key data settings in `configs/train.yaml`:
```yaml
horizons: [2, 5, 10]  # Forecast horizons in trading days

preprocessing:
  log_transform_cols: ["RV_fwd_{h}", "RV_back_{h}", "IV"]
  winsorize_cols:
    RET_SPY: [0.005, 0.005]  # Clip top/bottom 0.5%
```

---

## 7. Usage

```python
import pandas as pd

# Load processed data
df = pd.read_pickle('data/processed/timeseries.pkl')

# Access key columns
spy_returns = df['RET_SPY']
implied_vol = df['IV']

# Note: Forward RV targets (RV_fwd_2, RV_fwd_5, RV_fwd_10) and
# backward RV features (RV_back_2, RV_back_5, RV_back_10) are
# computed dynamically during training for each horizon.
```
