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

#### 2.2.1 Proof of No Lookahead Bias in Forward RV Computation

Let's trace through exactly what data is used at each step with a **concrete numerical example**:

**Test Setup:**
```
t=0: r[0] = 0.01
t=1: r[1] = -0.02
t=2: r[2] = 0.03
t=3: r[3] = 0.01
t=4: r[4] = -0.01
...
```

**Step-by-step trace for `compute_forward_realized_volatility(returns, h=3)`:**

| Step | Operation | At t=0 | At t=1 | At t=2 |
|------|-----------|--------|--------|--------|
| 1 | `shift(-1)` | future_returns[0] = r[1] = -0.02 | future_returns[1] = r[2] = 0.03 | future_returns[2] = r[3] = 0.01 |
| 2 | `rolling(3).std()` | NaN (not enough data) | NaN | std(r[1],r[2],r[3]) at index 2 |
| 3 | `shift(-2)` | Value from index 2 → std(r[1],r[2],r[3]) | Value from index 3 | Value from index 4 |

**Result at t=0:**
- `RV_fwd_3[t=0]` = $\sqrt{252} \times \text{std}(r_1, r_2, r_3)$ = $\sqrt{252} \times 0.02517$ = **0.3995** ✓

**Verification Code Output:**
```
Manual at t=0: std(r[1], r[2], r[3]) = 0.399500
Function at t=0: 0.399500
MATCH: True
```

**Conclusion**: At time $t$, `RV_fwd_h[t]` uses **only** returns $r_{t+1}, r_{t+2}, \ldots, r_{t+h}$ — all **strictly future** returns. The target definition is **correct** with no lookahead bias.

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

#### 2.3.1 Proof of No Lookahead Bias in Backward RV

**Step-by-step trace for `compute_backward_realized_volatility(returns, h=3)`:**

At index $t$, `rolling(3).std()` computes `std(returns[t-2], returns[t-1], returns[t])`.

**Verification with numerical example:**
```
t=2: RV_back_3 uses r[0], r[1], r[2] → std(0.01, -0.02, 0.03) × √252 = 0.3995
t=3: RV_back_3 uses r[1], r[2], r[3] → std(-0.02, 0.03, 0.01) × √252 = 0.3995
t=4: RV_back_3 uses r[2], r[3], r[4] → std(0.03, 0.01, -0.01) × √252 = 0.3175
```

**Conclusion**: The backward RV at index $t$ uses **only past and current returns** $r_{t-h+1}, \ldots, r_t$. This is safe to use as a feature. ✓

### 2.4 Implied Volatility (Feature)
The VIX index is quoted in annualized percentage points (e.g., 20.0). We convert this to a decimal format for consistency with RV:
$$ IV_t = \frac{VIX_t}{100} $$

**VIX Timing Semantics**: VIX at time $t$ represents the market's implied volatility as of the **close** of trading day $t$. It reflects all information available up to and including day $t$. Therefore, using $IV_t$ to predict $RV_{fwd,h}[t]$ (which uses future returns $r_{t+1}, \ldots, r_{t+h}$) is **valid** — no lookahead bias.

**Note on VIX vs. Realized Volatility Horizon**:
- VIX represents the market's expectation of **30-day** forward volatility.
- Our target horizons include **2, 5, 10, and 30 days**.
- For the 30-day horizon, VIX is a direct comparable benchmark.
- For shorter horizons, VIX still captures the overall volatility regime and is useful as a feature.

---

## 3. LSTM Data Alignment Verification

The LSTM model uses a sequence of past observations to predict future volatility. A critical question is whether the `TimeSeriesDataset` correctly aligns features and targets.

### 3.1 TimeSeriesDataset Implementation

```python
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        return self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len - 1]
```

### 3.2 Alignment Proof

For `idx=0` with `seq_len=3`:
- `X` sequence: `X[0:3]` = features at indices 0, 1, 2
- `y` target: `y[0 + 3 - 1]` = `y[2]` = target at index 2

**Interpretation**: To predict `y[2]` (which is `RV_fwd_h` at time $t=2$), the model uses features `X[0], X[1], X[2]` (features from times $t=0, 1, 2$).

Since:
- Features (`IV`, `RV_back_h`) at time $t$ use only data up to and including time $t$
- Target `RV_fwd_h[t=2]` uses returns $r_3, r_4, \ldots, r_{2+h}$ (strictly future)

**Conclusion**: The LSTM sees features up to time $t$ and predicts forward RV starting at $t+1$. **No lookahead bias.** ✓

---

## 4. Statistical Rigor & Diagnostics

We performed rigorous statistical testing to validate model assumptions, ensuring that our choice of GARCH specification is data-driven.

### 4.1 Stationarity (ADF & KPSS Tests)
Stationarity is a prerequisite for GARCH modeling.
*   **ADF Test**: $p < 0.0001$ (Reject $H_0$: Unit Root)
*   **KPSS Test**: $p = 0.1000$ (Fail to Reject $H_0$: Stationary)
*   **Conclusion**: Log returns are **stationary**.
*   **Visual Proof**: The returns plot in `02_volatility_clustering.png` shows a constant mean around zero, unlike the non-stationary price series in `01_market_overview.png`.

### 4.2 Volatility Clustering (ARCH-LM & Ljung-Box)
We verified the presence of "ARCH effects" (volatility clustering), which justifies the use of GARCH models.
*   **ARCH-LM Test**: $p < 0.0001$ (Statistic: 1389.1)
*   **Ljung-Box (Squared Returns)**: $p < 0.0001$ (Statistic: 3738.1 at lag 10)
*   **Conclusion**: Strong evidence of volatility clustering.
*   **Visual Proof**:
    *   `02_volatility_clustering.png`: Shows clear bursts of high volatility.
    *   `04_autocorrelation_check.png`: Squared returns show significant, persistent autocorrelation, while raw returns do not.

### 4.3 Normality & Fat Tails (Jarque-Bera)
Financial returns typically exhibit "fat tails" (leptokurtosis) and skewness.
*   **Jarque-Bera Test**: $p < 0.0001$ (Statistic: 24056.6)
*   **Skewness**: -0.1115 (Slightly negative, indicating crash risk)
*   **Excess Kurtosis**: 10.02 (Heavy tails)
*   **Conclusion**: Returns are **highly non-normal**. We must use a **Skewed Student's t-distribution** (`dist='skewt'`).
*   **Visual Proof**:
    *   `03_distribution_fat_tails.png`: The empirical distribution (gray) is much taller and narrower than the Normal fit (red), with fatter tails.
    *   `09_qq_plot.png`: The S-shape deviation from the red line confirms heavy tails.

### 4.4 Leverage Effect
We tested if negative returns increase volatility more than positive returns (the "Leverage Effect").
*   **Evidence**: Strong negative correlation between $r_t$ and $\sigma^2_{t+k}$.
*   **Model Selection**: Standard GARCH assumes symmetric response. We chose **EGARCH** (Exponential GARCH), which explicitly models this asymmetry.
*   **Visual Proof**: `05_leverage_effect.png` shows negative correlations at all lags.

### 4.5 Model Selection (Grid Search)
We ran a comprehensive grid search over GARCH and EGARCH models with Normal, Student-t, and Skewed Student-t distributions. We tested lag orders $p \in \{1, 2\}$ and $q \in \{1, 2\}$.

*   **Selection Criteria**: Lowest BIC *conditional* on passing the ARCH-LM test on residuals (ensuring no remaining heteroskedasticity).
*   **Winner**: **EGARCH(2,1) with Skewed Student's t-distribution**.
    *   BIC: 15401.9 (Lowest)
    *   Residual ARCH-LM p-value: 0.8386 (Passed)
    *   Residual Ljung-Box p-value: 0.8509 (Passed)
*   **Comparison**: The best standard GARCH model (GARCH(2,1) + skewt) had a significantly higher BIC (15638.5), indicating a worse fit.

### 4.6 GARCH vs. EGARCH: The Leverage Effect
To understand *why* EGARCH outperforms GARCH, we generated the **News Impact Curve (NIC)** (`10_garch_vs_egarch_nic.png`).
*   **What it shows**: How volatility at $t+1$ responds to a shock (return) at $t$.
*   **GARCH (Blue Dashed)**: Symmetric parabola. It assumes a +5% rally and a -5% crash increase volatility by the exact same amount.
*   **EGARCH (Red Solid)**: Asymmetric curve. It shows that negative shocks (crashes) increase volatility *much more* than positive shocks (rallies).
*   **Conclusion**: The EGARCH model correctly captures the "panic" dynamic of financial markets, where fear drives volatility higher than greed. This aligns with the leverage effect observed in Section 7.5.

---

## 5. Preprocessing Pipeline

### 5.1 Log Transformation
Volatility is strictly positive and log-normally distributed. For the LSTM, we log-transform both inputs and targets:
$$ y_{train} = \ln(RV_{target}) $$
$$ X_{train} = \ln(IV_{input}) $$
This stabilizes the variance and allows the model to learn relative errors rather than absolute ones.

### 5.2 Data Splitting (Temporal)
To respect the time-series nature of the data, we use strict temporal splitting (no shuffling):

| Split | Period | Purpose |
|-------|--------|---------|
| **Train** | 1993-01-01 to 2015-12-31 | Model parameter estimation |
| **Validation** | 2016-01-01 to 2019-12-31 | Hyperparameter tuning & Early Stopping |
| **Test** | 2020-01-01 to Present | Final Out-of-Sample Performance |

### 5.3 Target Leakage Prevention

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

## 6. Data Edge Handling

### 6.1 Forward RV at Data Boundaries

The processed data correctly handles the trailing edge. Since `RV_fwd_30` requires returns from days $t+1$ to $t+30$, the last 30 trading days of raw data cannot have valid forward RV values.

**Verification from actual data:**
```
Raw SPY ends at: 2025-11-26
Processed data ends at: 2025-10-15
Gap: ~30 trading days ✓
```

The `dropna()` in `build_and_save_streams` automatically removes rows where any forward RV is NaN, ensuring clean data boundaries.

---

## 7. Exploratory Data Analysis (EDA) & Visual Proofs

We generate a comprehensive suite of plots in `artifacts/plots/data_analysis/` to validate our data assumptions and understand market dynamics.

### 7.1 Market Overview (`01_market_overview.png`)
*   **What it is**: Time series of SPY price and VIX index.
*   **Why it's useful**: Provides context on market regimes (bull/bear markets) and volatility spikes.
*   **Interpretation**: Note the inverse relationship; VIX spikes during market crashes (2008, 2020).

### 7.2 Volatility Clustering (`02_volatility_clustering.png`)
*   **What it is**: Daily log returns and squared returns over time.
*   **Why it's useful**: Visualizes "ARCH effects" — the tendency for volatility to cluster.
*   **Interpretation**: Periods of high variance (large squared returns) tend to persist, justifying the use of GARCH/LSTM models which have memory.

### 7.3 Distribution Analysis (`03_distribution_fat_tails.png`)
*   **What it is**: Histogram of returns overlaid with Normal and Student-t distributions.
*   **Why it's useful**: Checks the Normality assumption.
*   **Interpretation**: The "tall peak" and "fat tails" (excess kurtosis) show that extreme events happen far more often than a Normal distribution predicts. This confirms we must use heavy-tailed distributions (like Student-t) for modeling.

### 7.4 Autocorrelation (`04_autocorrelation_check.png`)
*   **What it is**: ACF of raw returns vs. squared returns.
*   **Why it's useful**: Tests for serial correlation.
*   **Interpretation**:
    *   **Raw Returns**: No significant autocorrelation (Market Efficiency / Random Walk).
    *   **Squared Returns**: Strong, persistent autocorrelation. This confirms that while *direction* is hard to predict, *magnitude* (volatility) is highly predictable.

### 7.5 Leverage Effect (`05_leverage_effect.png`)
*   **What it is**: Correlation between returns at $t$ and volatility at $t+k$.
*   **Why it's useful**: Checks if negative returns impact volatility differently than positive ones.
*   **Interpretation**: Negative correlations indicate the "Leverage Effect": market drops cause panic (higher vol), while rallies decrease vol. This justifies using EGARCH (asymmetric) over standard GARCH.

### 7.6 RV vs. VIX & Risk Premium (`06_rv_vs_vix.png`)
*   **What it is**: Comparison of 30-day Realized Volatility (RV) and Implied Volatility (VIX), plus the spread (VIX - RV).
*   **Why it's useful**: Analyzes the "Volatility Risk Premium" (VRP).
*   **Interpretation**:
    *   **VIX > RV**: Usually, VIX trades above RV (Positive VRP), as investors pay a premium for options protection.
    *   **VIX < RV**: In extreme crashes, realized volatility can exceed implied volatility.
    *   **Tracking**: VIX is a strong predictor of future RV, making it a crucial feature.

### 7.7 RV Term Structure (`07_rv_horizons.png`)
*   **What it is**: Realized Volatility calculated over different horizons (e.g., 5-day vs. 30-day).
*   **Why it's useful**: Shows the noise level differences.
*   **Interpretation**: Short-term RV (5-day) is much noisier and spikes faster. Long-term RV (30-day) is smoother. Our models need to handle these different dynamics.

### 7.8 Feature Correlation (`08_correlation_matrix.png`)
*   **What it is**: Heatmap of correlations between Returns, VIX, and RV targets.
*   **Why it's useful**: Identifies multicollinearity and predictive relationships.
*   **Interpretation**:
    *   **VIX vs RV**: High positive correlation (good predictor).
    *   **Returns vs VIX/RV**: Negative correlation (Leverage effect).
    *   **RV_5 vs RV_30**: High correlation, implying persistence across time scales.

### 7.9 Q-Q Plot (`09_qq_plot.png`)
*   **What it is**: Quantile-Quantile plot of returns against a Normal distribution.
*   **Why it's useful**: A rigorous test for non-normality.
*   **Interpretation**:
    *   **Straight line**: Data is Normal.
    *   **S-shape (deviations at ends)**: Fat tails. Our plot shows significant deviation at the extremes, confirming that a Gaussian model would underestimate risk (VaR/ES).

---

## 8. Configuration

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
```

---

## 9. Data Pipeline Architecture

The data pipeline is cleanly separated into two stages:

### 9.1 Data Generation (`download_data.py` → `features.py`)
Run once to download and preprocess all data:
```bash
python scripts/download_data.py --config configs/data.yaml
```

This creates `data/processed/timeseries.pkl` containing:
- `RET_SPY`: Daily log returns
- `IV`: Implied volatility (VIX / 100)
- `RV_fwd_2`, `RV_fwd_5`, `RV_fwd_10`, `RV_fwd_30`: Forward realized volatility targets
- `RV_back_2`, `RV_back_5`, `RV_back_10`, `RV_back_30`: Backward realized volatility features

### 9.2 Training (`run_train.py` → `train_loop.py`)
Uses pre-computed data, applies preprocessing (log transform), and trains models:
```bash
python scripts/run_train.py --train_cfg configs/train.yaml --model_cfg configs/model/lstm_vix.yaml
```

**Key Design Principle**: All RV columns are computed during data generation, not during training. This ensures:
1. Consistent target definitions across all models
2. No duplicate code
3. Clear separation of concerns
4. Data can be inspected before training

---

## 10. Known Issues & Considerations

### 10.1 VIX Horizon Mismatch (for short horizons)
VIX measures 30-day implied volatility. For horizons h < 30, this is not a perfect predictor, but it still provides valuable regime information. The 30-day horizon now allows direct VIX vs RV comparison.

### 10.2 LSTM Sequence Handling
The LSTM model uses a sequence of past observations to predict future volatility. The `TimeSeriesDataset` aligns predictions to the **last** timestamp in each sequence window:
```python
# TimeSeriesDataset.__getitem__:
return self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len - 1]
```
This means the prediction at index `t` uses features from `t - seq_len + 1` to `t`, and the target is `RV_fwd_h[t]` (volatility from `t+1` to `t+h`). ✓

---

## 11. Summary of Lookahead Bias Analysis

| Component | Status | Verification |
|-----------|--------|--------------|
| Forward RV computation | ✓ Correct | Numerical test matches manual calculation |
| Backward RV computation | ✓ Correct | Uses only past/current data |
| VIX/IV feature | ✓ Correct | Close-of-day value available at prediction time |
| LSTM data alignment | ✓ Correct | Features at t predict target at t (which uses t+1...t+h) |
| Train/Val/Test split | ✓ Correct | Temporal split with gap for target leakage |

---

## 12. Usage

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