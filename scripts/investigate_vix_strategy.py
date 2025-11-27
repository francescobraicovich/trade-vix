import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from volatility_modelling.data.splits import split_by_date, DateSplits
import yaml

# Load data
df = pd.read_pickle("data/processed/timeseries.pkl")

# Load config
with open("configs/train.yaml", 'r') as f:
    train_cfg = yaml.safe_load(f)

splits_cfg = train_cfg["splits"]
splits = DateSplits(
    train_end=splits_cfg["train_end"],
    valid_end=splits_cfg["valid_end"],
    test_end=splits_cfg["test_end"]
)

train_df, val_df, test_df = split_by_date(df, splits)

# Use test set
df_test = test_df
spy_ret = df_test['RET_SPY']
vix = df_test['IV']
rv_fwd_30 = df_test['RV_fwd_30']

print("="*80)
print("VIX-BASED STRATEGY DEEP DIVE")
print("="*80)

# Compute variance swap P&L
var_swap_pnl = (vix**2 - rv_fwd_30**2).dropna()

print(f"\n1. VARIANCE SWAP P&L STATISTICS")
print(f"   Sample size: {len(var_swap_pnl)}")
print(f"   Mean: {var_swap_pnl.mean():.6f}")
print(f"   Std: {var_swap_pnl.std():.6f}")
print(f"   Min: {var_swap_pnl.min():.6f}")
print(f"   Max: {var_swap_pnl.max():.6f}")
print(f"   Skewness: {var_swap_pnl.skew():.4f}")
print(f"   Positive days: {(var_swap_pnl > 0).sum()} ({100*(var_swap_pnl > 0).sum()/len(var_swap_pnl):.1f}%)")
print(f"   Negative days: {(var_swap_pnl < 0).sum()} ({100*(var_swap_pnl < 0).sum()/len(var_swap_pnl):.1f}%)")

# Non-overlapping periods
vrp_horizon = 30
non_overlap_pnl = var_swap_pnl.iloc[::vrp_horizon]

print(f"\n2. NON-OVERLAPPING 30-DAY PERIODS")
print(f"   Number of periods: {len(non_overlap_pnl)}")
print(f"   Mean: {non_overlap_pnl.mean():.6f}")
print(f"   Std: {non_overlap_pnl.std():.6f}")
print(f"   Positive periods: {(non_overlap_pnl > 0).sum()} ({100*(non_overlap_pnl > 0).sum()/len(non_overlap_pnl):.1f}%)")

# VIX threshold computation - CRITICAL CHECK
print(f"\n3. VIX THRESHOLD COMPUTATION")
print(f"   Full VIX series length: {len(vix)}")
print(f"   VIX mean: {vix.mean():.4f}")
print(f"   VIX median: {vix.median():.4f}")
print(f"   VIX 70th percentile: {vix.quantile(0.70):.4f}")

# CHECK FOR LOOKAHEAD BIAS
print(f"\n4. LOOKAHEAD BIAS CHECK")
print(f"   Computing threshold on: Test set ({len(vix)} days)")
print(f"   ⚠️  WARNING: This is computed on the ENTIRE TEST SET!")
print(f"   ⚠️  This means we're using FUTURE information to set the threshold!")

# Proper expanding window threshold
vix_expanding_70th = vix.expanding(min_periods=252).quantile(0.70)
print(f"\n   If we use EXPANDING WINDOW (no lookahead):")
print(f"   First valid threshold: {vix_expanding_70th.dropna().iloc[0]:.4f}")
print(f"   Last threshold: {vix_expanding_70th.iloc[-1]:.4f}")
print(f"   Mean threshold: {vix_expanding_70th.mean():.4f}")

# Compare filtering
vix_threshold_fixed = vix.quantile(0.70)
vix_high_fixed = (vix > vix_threshold_fixed)

vix_high_expanding = (vix > vix_expanding_70th).fillna(False)

print(f"\n5. FILTERING COMPARISON")
print(f"   Fixed threshold (lookahead): {vix_high_fixed.sum()} days ({100*vix_high_fixed.sum()/len(vix):.1f}%)")
print(f"   Expanding threshold (no lookahead): {vix_high_expanding.sum()} days ({100*vix_high_expanding.sum()/len(vix):.1f}%)")

# Apply to non-overlapping periods
vix_periodic_fixed = vix.iloc[::vrp_horizon].reindex(non_overlap_pnl.index)
vix_high_periodic_fixed = (vix_periodic_fixed > vix_threshold_fixed)

vix_periodic_expanding = vix_expanding_70th.iloc[::vrp_horizon].reindex(non_overlap_pnl.index)
vix_high_periodic_expanding = (vix_periodic_fixed > vix_periodic_expanding)

print(f"\n6. PERIOD-LEVEL FILTERING")
print(f"   Fixed threshold: {vix_high_periodic_fixed.sum()} periods ({100*vix_high_periodic_fixed.sum()/len(non_overlap_pnl):.1f}%)")
print(f"   Expanding threshold: {vix_high_periodic_expanding.sum()} periods ({100*vix_high_periodic_expanding.sum()/len(non_overlap_pnl):.1f}%)")

# Scaling
raw_std = non_overlap_pnl.std()
target_annual_vol = 0.10
periods_per_year = 252 / vrp_horizon
target_period_std = target_annual_vol / np.sqrt(periods_per_year)
scale = target_period_std / raw_std

print(f"\n7. SCALING FACTOR")
print(f"   Raw std (30-day periods): {raw_std:.6f}")
print(f"   Target std: {target_period_std:.6f}")
print(f"   Scale factor: {scale:.4f}")

# Apply scaling
scaled_pnl = non_overlap_pnl * scale

# Strategy returns - FIXED THRESHOLD (CURRENT IMPLEMENTATION)
filtered_pnl_fixed = scaled_pnl.copy()
filtered_pnl_fixed[~vix_high_periodic_fixed] = 0

print(f"\n8. STRATEGY PERFORMANCE - FIXED THRESHOLD (CURRENT - HAS LOOKAHEAD)")
print(f"   Periods traded: {(filtered_pnl_fixed != 0).sum()}")
print(f"   Mean return per period: {filtered_pnl_fixed[filtered_pnl_fixed != 0].mean():.6f}")
print(f"   Win rate: {(filtered_pnl_fixed > 0).sum() / (filtered_pnl_fixed != 0).sum():.2%}")

# Convert to daily
daily_pnl_fixed = filtered_pnl_fixed.reindex(df_test.index, method='ffill').fillna(0)

# Calculate metrics
total_ret_fixed = (1 + daily_pnl_fixed).prod() - 1
ann_ret_fixed = (1 + daily_pnl_fixed).prod() ** (252 / len(daily_pnl_fixed)) - 1
ann_vol_fixed = daily_pnl_fixed.std() * np.sqrt(252)
sharpe_fixed = ann_ret_fixed / ann_vol_fixed if ann_vol_fixed > 0 else 0

print(f"   Annualized Return: {ann_ret_fixed:.2%}")
print(f"   Annualized Vol: {ann_vol_fixed:.2%}")
print(f"   Sharpe Ratio: {sharpe_fixed:.2f}")

# Strategy returns - EXPANDING THRESHOLD (CORRECT IMPLEMENTATION)
filtered_pnl_expanding = scaled_pnl.copy()
filtered_pnl_expanding[~vix_high_periodic_expanding] = 0

print(f"\n9. STRATEGY PERFORMANCE - EXPANDING THRESHOLD (CORRECT - NO LOOKAHEAD)")
print(f"   Periods traded: {(filtered_pnl_expanding != 0).sum()}")
if (filtered_pnl_expanding != 0).sum() > 0:
    print(f"   Mean return per period: {filtered_pnl_expanding[filtered_pnl_expanding != 0].mean():.6f}")
    print(f"   Win rate: {(filtered_pnl_expanding > 0).sum() / (filtered_pnl_expanding != 0).sum():.2%}")

# Convert to daily
daily_pnl_expanding = filtered_pnl_expanding.reindex(df_test.index, method='ffill').fillna(0)

# Calculate metrics
total_ret_expanding = (1 + daily_pnl_expanding).prod() - 1
ann_ret_expanding = (1 + daily_pnl_expanding).prod() ** (252 / len(daily_pnl_expanding)) - 1
ann_vol_expanding = daily_pnl_expanding.std() * np.sqrt(252)
sharpe_expanding = ann_ret_expanding / ann_vol_expanding if ann_vol_expanding > 0 else 0

print(f"   Annualized Return: {ann_ret_expanding:.2%}")
print(f"   Annualized Vol: {ann_vol_expanding:.2%}")
print(f"   Sharpe Ratio: {sharpe_expanding:.2f}")

# Visualizations
fig, axes = plt.subplots(4, 1, figsize=(14, 16))

# Plot 1: VIX over time with thresholds
ax = axes[0]
ax.plot(vix.index, vix.values, label='VIX', linewidth=1, alpha=0.7)
ax.axhline(vix_threshold_fixed, color='red', linestyle='--', label=f'Fixed 70th %ile: {vix_threshold_fixed:.4f}', linewidth=2)
ax.plot(vix_expanding_70th.index, vix_expanding_70th.values, color='green', linestyle='--', 
        label='Expanding 70th %ile', linewidth=2, alpha=0.7)
ax.fill_between(vix.index, 0, 1, where=vix_high_fixed, alpha=0.2, color='red', 
                label='Trade periods (Fixed)', transform=ax.get_xaxis_transform())
ax.set_title('VIX Levels and Threshold Comparison', fontsize=12, fontweight='bold')
ax.set_ylabel('VIX Level')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Plot 2: Variance Swap P&L
ax = axes[1]
ax.plot(var_swap_pnl.index, var_swap_pnl.values, linewidth=0.5, alpha=0.7)
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.fill_between(var_swap_pnl.index, 0, var_swap_pnl.values, where=(var_swap_pnl > 0), 
                alpha=0.3, color='green', label='Positive VRP')
ax.fill_between(var_swap_pnl.index, 0, var_swap_pnl.values, where=(var_swap_pnl < 0), 
                alpha=0.3, color='red', label='Negative VRP')
ax.set_title('Daily Variance Swap P&L (IV² - RV²)', fontsize=12, fontweight='bold')
ax.set_ylabel('P&L')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Plot 3: Equity curves comparison
ax = axes[2]
equity_fixed = (1 + daily_pnl_fixed).cumprod()
equity_expanding = (1 + daily_pnl_expanding).cumprod()

ax.plot(equity_fixed.index, equity_fixed.values, label=f'Fixed Threshold (Sharpe: {sharpe_fixed:.2f})', 
        linewidth=2, color='red')
ax.plot(equity_expanding.index, equity_expanding.values, 
        label=f'Expanding Threshold (Sharpe: {sharpe_expanding:.2f})', 
        linewidth=2, color='green')
ax.set_title('Equity Curve Comparison: Fixed vs Expanding Threshold', fontsize=12, fontweight='bold')
ax.set_ylabel('Growth of $1')
ax.set_yscale('log')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Plot 4: Rolling Sharpe Ratio (252-day window)
ax = axes[3]
rolling_window = 252
rolling_sharpe_fixed = (daily_pnl_fixed.rolling(rolling_window).mean() * 252) / (daily_pnl_fixed.rolling(rolling_window).std() * np.sqrt(252))
rolling_sharpe_expanding = (daily_pnl_expanding.rolling(rolling_window).mean() * 252) / (daily_pnl_expanding.rolling(rolling_window).std() * np.sqrt(252))

ax.plot(rolling_sharpe_fixed.index, rolling_sharpe_fixed.values, 
        label='Fixed Threshold', linewidth=2, color='red', alpha=0.7)
ax.plot(rolling_sharpe_expanding.index, rolling_sharpe_expanding.values, 
        label='Expanding Threshold', linewidth=2, color='green', alpha=0.7)
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.set_title('Rolling 1-Year Sharpe Ratio', fontsize=12, fontweight='bold')
ax.set_ylabel('Sharpe Ratio')
ax.set_xlabel('Date')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("artifacts/plots/vix_strategy_investigation.png", dpi=150)
print(f"\n✓ Saved plot to artifacts/plots/vix_strategy_investigation.png")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\n🚨 LOOKAHEAD BIAS DETECTED!")
print(f"\nThe VIX-Based strategy computes the 70th percentile threshold using the")
print(f"ENTIRE test set: vix.quantile(0.70)")
print(f"\nThis means:")
print(f"1. We know the future distribution of VIX when making trades")
print(f"2. The threshold ({vix_threshold_fixed:.4f}) is computed with hindsight")
print(f"3. The extraordinarily high Sharpe ratio ({sharpe_fixed:.2f}) is INVALID")
print(f"\nWith proper expanding window (no lookahead):")
print(f"- Sharpe Ratio: {sharpe_fixed:.2f} → {sharpe_expanding:.2f}")
print(f"- Annualized Return: {ann_ret_fixed:.2%} → {ann_ret_expanding:.2%}")
print(f"\nThe strategy performance degrades significantly when implemented correctly.")
