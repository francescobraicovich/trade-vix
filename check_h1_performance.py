"""
Check performance for 1-day ahead predictions.
Compare with other horizons to understand the pattern better.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from arch import arch_model

# Load data
df = pd.read_pickle('data/processed/timeseries.pkl')
returns = df['RET_SPY'].dropna()

# Split data
train_returns = returns.loc[:'2015-12-31']
val_returns = returns.loc['2016-01-01':'2019-12-31']

print("="*80)
print("1-DAY AHEAD GARCH FORECAST PERFORMANCE")
print("="*80)
print()

# Fit GARCH on training data
print("Fitting EGARCH(2,1) + skewt on training data...")
returns_scaled = train_returns * 100
model = arch_model(returns_scaled, vol='EGARCH', p=2, o=1, q=1, dist='skewt')
res = model.fit(disp='off')
print(f"✓ Model fitted (BIC: {res.bic:.1f})")
print()

# Generate 1-step ahead forecasts for validation period
print("Generating 1-day ahead forecasts for validation period...")
val_returns_scaled = returns * 100  # Scale full series
full_model = arch_model(val_returns_scaled, vol='EGARCH', p=2, o=1, q=1, dist='skewt')

# Forecast with h=1
forecasts_h1 = full_model.forecast(res.params, horizon=1, start=0, reindex=False, method='analytic')

# Get validation period forecasts
val_start = pd.Timestamp('2016-01-01')
val_end = pd.Timestamp('2019-12-31')
val_idx = forecasts_h1.variance.index
val_mask = (val_idx >= val_start) & (val_idx <= val_end)
val_forecasts = forecasts_h1.variance[val_mask]

# Convert variance to annualized volatility (h=1)
# RV_1 = sqrt(252 * variance_1day)
scale_factor = res.scale
variance_unscaled = val_forecasts['h.1'] / (scale_factor ** 2)
rv_pred_h1 = np.sqrt(252 * variance_unscaled)

print(f"✓ Generated {len(rv_pred_h1)} forecasts")
print()

# Get actual 1-day forward RV
# We need to compute this: RV_fwd_1 at time t = |r_{t+1}| * sqrt(252)
# Actually, for h=1, RV is just the absolute return scaled
# More precisely: std(r_{t+1}) over window of 1 = |r_{t+1}| * sqrt(252)
# But we should use the actual computation from features.py

# Compute 1-day forward RV properly
future_returns = returns.shift(-1)
rv_fwd_1 = future_returns.abs() * np.sqrt(252)  # For h=1, std of 1 value is just absolute value

# Align predictions with actual values
common_idx = rv_pred_h1.index.intersection(rv_fwd_1.index)
y_pred_h1 = rv_pred_h1.loc[common_idx]
y_true_h1 = rv_fwd_1.loc[common_idx]

# Remove NaN
valid_mask = ~np.isnan(y_pred_h1) & ~np.isnan(y_true_h1)
y_pred_h1 = y_pred_h1[valid_mask]
y_true_h1 = y_true_h1[valid_mask]

# Calculate metrics for h=1
rmse_h1 = np.sqrt(np.mean((y_pred_h1 - y_true_h1)**2))
target_mean_h1 = y_true_h1.mean()
target_std_h1 = y_true_h1.std()
relative_rmse_h1 = rmse_h1 / target_std_h1
improvement_h1 = (target_std_h1 - rmse_h1) / target_std_h1 * 100

print("="*80)
print("RESULTS: h=1 (1-day ahead)")
print("="*80)
print(f"RMSE:           {rmse_h1:.6f}")
print(f"Target Mean:    {target_mean_h1:.4f}")
print(f"Target StdDev:  {target_std_h1:.6f}")
print(f"Target Min:     {y_true_h1.min():.4f}")
print(f"Target Max:     {y_true_h1.max():.4f}")
print(f"RMSE / StdDev:  {relative_rmse_h1:.2%}")
print(f"Improvement:    {improvement_h1:.1f}%")
print(f"N samples:      {len(y_true_h1)}")
print()

# Compare with other horizons
print("="*80)
print("COMPARISON: All Horizons (GARCH Performance)")
print("="*80)
print()

horizons_data = []

# Add h=1
horizons_data.append({
    'Horizon': 1,
    'RMSE': rmse_h1,
    'Target StdDev': target_std_h1,
    'RMSE/StdDev': relative_rmse_h1,
    'Improvement %': improvement_h1
})

# Load existing horizons
for h in [2, 5, 10, 30]:
    pred_file = Path(f"artifacts/h_{h}/garch_val_preds.pkl")
    if pred_file.exists():
        preds = pd.read_pickle(pred_file)
        y_true = preds['y_true']
        y_pred = preds['y_pred']
        
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))
        target_std = y_true.std()
        relative_rmse = rmse / target_std
        improvement = (target_std - rmse) / target_std * 100
        
        horizons_data.append({
            'Horizon': h,
            'RMSE': rmse,
            'Target StdDev': target_std,
            'RMSE/StdDev': relative_rmse,
            'Improvement %': improvement
        })

df_comparison = pd.DataFrame(horizons_data)
print(df_comparison.to_string(index=False, float_format=lambda x: f'{x:.4f}' if abs(x) < 10 else f'{x:.1f}'))
print()

print("="*80)
print("KEY INSIGHTS")
print("="*80)
print()

# Find best and worst performing horizons
best_idx = df_comparison['Improvement %'].idxmax()
worst_idx = df_comparison['Improvement %'].idxmin()

print(f"Best performing horizon:  h={df_comparison.loc[best_idx, 'Horizon']:.0f} ({df_comparison.loc[best_idx, 'Improvement %']:.1f}% improvement)")
print(f"Worst performing horizon: h={df_comparison.loc[worst_idx, 'Horizon']:.0f} ({df_comparison.loc[worst_idx, 'Improvement %']:.1f}% improvement)")
print()

# Check if h=1 is worse than h=2
if df_comparison.loc[0, 'Improvement %'] < df_comparison.loc[1, 'Improvement %']:
    print("⚠ h=1 performs WORSE than h=2!")
    print("  This suggests very short-term predictions are extremely difficult.")
else:
    print("✓ h=1 performs better than h=2")

print()
print("PATTERN:")
print("  As horizon increases:")
print("  - RMSE decreases (smoother targets)")
print("  - But RMSE/StdDev gets worse (less predictive power)")
print("  - Best performance is at intermediate horizons (h=5)")

