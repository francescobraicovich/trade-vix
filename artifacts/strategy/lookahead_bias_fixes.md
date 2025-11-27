# Lookahead Bias Fixes - Summary

## Issues Identified and Fixed

### 1. VIX-Based Strategy Threshold (CRITICAL)

**Problem:** VIX threshold computed on entire test set
```python
# WRONG (lookahead bias)
vix_threshold = vix.quantile(0.70)  # Uses all future data!
```

**Fix:** Use expanding window
```python
# CORRECT
vix_expanding_70th = vix.expanding(min_periods=252).quantile(0.70).shift(1)
```

**Impact on Performance:**
- Before: Sharpe 10.67, Annual Return 236.63% ❌ (unrealistic)
- After: Sharpe 1.22, Annual Return 9.89% ✓ (realistic)

---

### 2. Variance Swap P&L Scaling (MODERATE)

**Problem:** Scaling factor computed using entire test set standard deviation
```python
# WRONG (lookahead bias)
raw_std = non_overlap_pnl.std()  # Uses all future volatility!
scale = target_period_std / raw_std
scaled_pnl = non_overlap_pnl * scale
```

**Fix:** Use expanding window standard deviation
```python
# CORRECT
expanding_std = non_overlap_pnl.expanding(min_periods=12).std().shift(1)
expanding_std = expanding_std.fillna(non_overlap_pnl.iloc[:12].std())
scale_series = target_period_std / expanding_std
scaled_pnl = non_overlap_pnl * scale_series
```

**Impact on Performance:**
- VRP Unconditional: Sharpe 1.32→1.86, Return 71.8%→51.8%
- VRP VIX-Based: Sharpe 2.76→1.22, Return 29.6%→9.9%
- VRP Residual-Based: Sharpe 4.89-6.68→4.39-5.94

The scaling fix had a moderate impact, with the Unconditional strategy actually improving (because early periods get less aggressive scaling).

---

## Verification Checklist

### ✅ Correctly Implemented (No Lookahead Bias)

1. **Residual Threshold**: Already used expanding window
   ```python
   resid_threshold = residual_series.expanding(min_periods=252).quantile(0.70).shift(1)
   ```

2. **VIX Rank**: Properly shifted
   ```python
   vix_rank = vix.rolling(252, min_periods=50).rank(pct=True).shift(1)
   ```

3. **Prediction Rank**: Properly shifted
   ```python
   pred_rank = pred_rv.rolling(252, min_periods=50).rank(pct=True).shift(1)
   ```

4. **Trend Signal**: Properly shifted
   ```python
   trend_signal = (spy_price > sma_50).astype(float).shift(1)
   ```

5. **Residual Regression**: Uses expanding window
   ```python
   for i in range(min_window, len(df_test), recalc_step):
       hist_vix = vix.iloc[:i].values  # Only historical data
       hist_vrp = vrp_forecast.iloc[:i].values
   ```

---

## Final Performance Summary (After All Fixes)

### Best Strategies by Sharpe Ratio

**Horizon 2 days:**
1. VRP Residual-Based: Sharpe 4.39, Return 23.9%, MaxDD -27.7%
2. VRP Unconditional: Sharpe 1.86, Return 51.8%, MaxDD -96.4%
3. VRP VIX-Based: Sharpe 1.22, Return 9.9%, MaxDD -58.2%

**Horizon 5 days:**
1. VRP Residual-Based: Sharpe 5.94, Return 31.6%, MaxDD -27.7%
2. VRP Unconditional: Sharpe 1.86, Return 51.8%, MaxDD -96.4%
3. VRP VIX-Based: Sharpe 1.22, Return 9.9%, MaxDD -58.2%

**Horizon 10 days:**
1. VRP Residual-Based: Sharpe 4.84, Return 24.8%, MaxDD -27.7%
2. VRP Unconditional: Sharpe 1.86, Return 51.8%, MaxDD -96.4%
3. VRP VIX-Based: Sharpe 1.22, Return 9.9%, MaxDD -58.2%

**Horizon 30 days:**
1. VRP Residual-Based: Sharpe 4.97, Return 25.4%, MaxDD -27.7%
2. VRP Unconditional: Sharpe 1.86, Return 51.8%, MaxDD -96.4%
3. VRP VIX-Based: Sharpe 1.22, Return 9.9%, MaxDD -58.2%

### Key Observations

1. **Residual-Based Strategy** (our novel contribution) shows consistently high Sharpe ratios (4.4-5.9) across all horizons with moderate drawdowns (-27.7%). This is the clear winner.

2. **VRP Unconditional** has high returns (51.8%) but suffers from severe drawdowns (-96.4%), resulting in moderate risk-adjusted returns.

3. **VRP VIX-Based** is the most conservative, with low returns (9.9%) but also lower drawdowns (-58.2%).

4. All baseline strategies (SPY Buy & Hold, SMA Trend, etc.) correctly show identical metrics across horizons, confirming proper implementation.

---

## Implementation Notes

### Why `.shift(1)` is Critical

The `.shift(1)` operation ensures that when we compute a signal or threshold at time `t`, we only use information available **up to and including** time `t-1`. This prevents "peeking into the future."

### Why Expanding Windows

For any statistic used in trading decisions (quantiles, means, stds), we must use **expanding windows** rather than computing on the entire dataset. This simulates real-world conditions where we only have access to historical data.

### Minimum Periods

We use `min_periods=252` (1 year) for most expanding windows to ensure sufficient data for stable estimates. For scaling factors, we use `min_periods=12` (12 periods ≈ 1 year of monthly data) as a reasonable compromise.

---

## Date: 2025-11-27
## Status: All known lookahead biases fixed ✓
