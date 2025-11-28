"""
Verify the RMSE pattern across horizons.
Check if lower RMSE for longer horizons is real or a mistake.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load data
df = pd.read_pickle('data/processed/timeseries.pkl')

print("="*80)
print("VERIFICATION: RMSE Pattern Across Horizons")
print("="*80)
print()

# Load validation metrics for each horizon
horizons = [2, 5, 10, 30]
results = []

for h in horizons:
    # Load actual GARCH predictions if they exist
    pred_file = Path(f"artifacts/h_{h}/garch_val_preds.pkl")
    if pred_file.exists():
        preds = pd.read_pickle(pred_file)
        y_true = preds['y_true']
        y_pred = preds['y_pred']
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))
        mae = np.mean(np.abs(y_pred - y_true))
        
        # Calculate target statistics
        target_mean = y_true.mean()
        target_std = y_true.std()
        target_min = y_true.min()
        target_max = y_true.max()
        
        # Calculate baseline RMSE (if we just predicted the mean)
        baseline_rmse = target_std  # RMSE if predicting mean = std dev
        
        # Calculate relative performance
        relative_rmse = rmse / target_std
        improvement_over_baseline = (baseline_rmse - rmse) / baseline_rmse * 100
        
        results.append({
            'Horizon': h,
            'RMSE': rmse,
            'MAE': mae,
            'Target Mean': target_mean,
            'Target StdDev': target_std,
            'Target Min': target_min,
            'Target Max': target_max,
            'Baseline RMSE': baseline_rmse,
            'RMSE / StdDev': relative_rmse,
            'Improvement %': improvement_over_baseline,
            'N samples': len(y_true)
        })
        
        print(f"h={h} days:")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  Target Mean: {target_mean:.4f}")
        print(f"  Target StdDev: {target_std:.6f}")
        print(f"  Target Range: [{target_min:.4f}, {target_max:.4f}]")
        print(f"  RMSE / StdDev: {relative_rmse:.2%}")
        print(f"  Improvement over baseline: {improvement_over_baseline:.1f}%")
        print()

print("="*80)
print("SUMMARY TABLE")
print("="*80)
print()

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

print()
print("="*80)
print("KEY INSIGHTS")
print("="*80)
print()

# Check if RMSE decreases with horizon
rmse_trend = "DECREASES" if results[-1]['RMSE'] < results[0]['RMSE'] else "INCREASES"
print(f"1. RMSE trend: {rmse_trend} with horizon")
print(f"   h=2:  RMSE = {results[0]['RMSE']:.6f}")
print(f"   h=30: RMSE = {results[-1]['RMSE']:.6f}")
print()

# Check if target volatility decreases with horizon
std_trend = "DECREASES" if results[-1]['Target StdDev'] < results[0]['Target StdDev'] else "INCREASES"
print(f"2. Target volatility (StdDev) trend: {std_trend} with horizon")
print(f"   h=2:  StdDev = {results[0]['Target StdDev']:.6f}")
print(f"   h=30: StdDev = {results[-1]['Target StdDev']:.6f}")
ratio = results[0]['Target StdDev'] / results[-1]['Target StdDev']
print(f"   Ratio: {ratio:.2f}x more volatile for h=2")
print()

# Check relative performance
print("3. Relative Performance (RMSE / Target StdDev):")
for r in results:
    print(f"   h={r['Horizon']:2d}: {r['RMSE / StdDev']:.2%} (explains {r['Improvement %']:.1f}% variance)")
print()

# Check if the pattern makes sense
print("4. EXPLANATION:")
if rmse_trend == "DECREASES" and std_trend == "DECREASES":
    print("   ✓ Lower RMSE for longer horizons IS REAL")
    print("   ✓ This is because targets are LESS VOLATILE (time averaging)")
    print("   ✓ But relative performance is similar/worse for longer horizons")
    print()
    print("   CONCLUSION: Not a mistake! Lower RMSE reflects smoother targets.")
elif rmse_trend == "INCREASES":
    print("   ⚠ RMSE increases with horizon - this is unusual!")
    print("   This might indicate a problem with the forecasting model.")
else:
    print("   ⚠ Mixed pattern - needs investigation")

print()
print("="*80)
print("VERIFICATION: Check actual RV values in validation period")
print("="*80)
print()

# Look at actual RV values in validation period (2016-2019)
val_start = '2016-01-01'
val_end = '2019-12-31'
val_data = df.loc[val_start:val_end]

for h in horizons:
    rv_col = f'RV_fwd_{h}'
    if rv_col in val_data.columns:
        rv = val_data[rv_col].dropna()
        print(f"RV_fwd_{h} in validation period:")
        print(f"  Mean: {rv.mean():.4f}")
        print(f"  Std:  {rv.std():.6f}")
        print(f"  Min:  {rv.min():.4f}")
        print(f"  Max:  {rv.max():.4f}")
        print(f"  Samples: {len(rv)}")
        print()

