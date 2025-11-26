import os
import sys
import shutil
import subprocess
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_command(command, description):
    print(f"\n=== {description} ===")
    print(f"Running: {command}")
    try:
        subprocess.run(command, shell=True, check=True, executable='/bin/zsh')
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def cleanup_artifacts():
    print("\n=== Cleaning Artifacts ===")
    artifacts_dir = Path("artifacts")
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    artifacts_dir.mkdir()
    (artifacts_dir / "checkpoints").mkdir()
    (artifacts_dir / "diagnostics").mkdir()
    (artifacts_dir / "strategy").mkdir()
    (artifacts_dir / "plots").mkdir()
    print("Artifacts directory cleaned and recreated.")

def run_training_pipeline():
    # 1. GARCH Grid Search & Diagnostics
    run_command(
        "python scripts/garch_diagnostics.py",
        "GARCH Grid Search & Diagnostics"
    )
    
    # 2. Train Models
    # We need to train for each horizon? 
    # run_train.py handles multiple horizons if specified in config.
    # Let's check train.yaml to see if it has [2, 5, 10].
    
    # Train GARCH
    run_command(
        "python scripts/run_train.py --train_cfg configs/train.yaml --model_cfg configs/model/garch.yaml",
        "Training GARCH"
    )
    
    # Train LSTM-RV
    run_command(
        "python scripts/run_train.py --train_cfg configs/train.yaml --model_cfg configs/model/lstm_rv.yaml",
        "Training LSTM-RV"
    )
    
    # Train LSTM-VIX
    run_command(
        "python scripts/run_train.py --train_cfg configs/train.yaml --model_cfg configs/model/lstm_vix.yaml",
        "Training LSTM-VIX"
    )

def load_predictions(horizons=[2, 5, 10]):
    preds = {}
    models = ['garch', 'lstm_rv', 'lstm_vix']
    splits = ['train', 'val', 'test']
    
    base_dir = Path("artifacts")
    
    for h in horizons:
        for model in models:
            dfs = []
            for split in splits:
                # Path structure from run_train.py: artifacts/h_{h}/{model}_{split}_preds.pkl
                # Wait, run_train.py saves to artifacts_dir/h_{h}/...
                # Let's verify where run_train.py saves.
                # It uses train_cfg["paths"]["artifacts_dir"] which is usually "artifacts"
                # So artifacts/h_{h}/...
                
                # But wait, run_train.py doesn't include model name in folder, 
                # it saves as {model_name}_{split}_preds.pkl inside h_{h}
                
                p = base_dir / f"h_{h}" / f"{model}_{split}_preds.pkl"
                if p.exists():
                    data = pd.read_pickle(p)
                    if isinstance(data, dict):
                        dfs.append(data['y_pred'])
                    else:
                        dfs.append(data)
            
            if dfs:
                full_df = pd.concat(dfs).sort_index()
                preds[f"{model}_h{h}"] = full_df
                
    return pd.DataFrame(preds)

def calculate_metrics(returns, benchmark=None):
    # Annualized metrics
    ann_factor = 252
    
    total_ret = (1 + returns).prod() - 1
    ann_ret = (1 + returns).prod() ** (ann_factor / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
    
    # Max Drawdown
    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()
    
    metrics = {
        "Total Return": total_ret,
        "Annualized Return": ann_ret,
        "Annualized Vol": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd
    }
    
    if benchmark is not None:
        # Alpha / Beta
        # Align
        common = returns.index.intersection(benchmark.index)
        y = returns.loc[common]
        X = benchmark.loc[common]
        
        if len(y) > 0:
            slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
            metrics["Alpha"] = intercept * ann_factor
            metrics["Beta"] = slope
            metrics["Correlation"] = r_value
            
    return metrics

def run_strategies_and_backtest():
    """
    Implement all strategies from strategies.md:
    
    EQUITY STRATEGIES (Long SPY):
    1. Buy & Hold
    2. SMA(50) Trend Only
    3. Trend + VIX Sizing
    4. Trend + Prediction Sizing
    
    VRP STRATEGIES (Variance Swap - Sell Volatility):
    5. Unconditional (Always Sell)
    6. VIX-Based Filtering
    7. Residual-Based (Our Unique Alpha)
    """
    print("\n=== Running Backtests ===")
    
    # Load Data
    df = pd.read_pickle("data/processed/timeseries.pkl")
    
    # Load Predictions
    preds = load_predictions()
    if preds.empty:
        print("WARNING: No predictions found! Will skip prediction-based strategies.")
        preds = pd.DataFrame(index=df.index)
    
    # Align everything
    common_idx = df.index.intersection(preds.index) if not preds.empty else df.index
    df = df.loc[common_idx]
    if not preds.empty:
        preds = preds.loc[common_idx]
    
    spy_ret = df['RET_SPY']
    vix = df['IV']  # VIX in decimal (0.20 = 20%)
    
    # Build ensemble prediction (same weights as in analysis)
    if 'lstm_vix_h5' in preds.columns and 'garch_h5' in preds.columns:
        # Predictions are in raw scale (annualized vol).
        # We combine them in log-space (geometric mean) as per original analysis weights.
        log_lstm = np.log(preds['lstm_vix_h5'])
        log_garch = np.log(preds['garch_h5'])
        pred_rv = np.exp(log_lstm * 0.64 + log_garch * 0.36)
    elif 'garch_h5' in preds.columns:
        pred_rv = preds['garch_h5']
    else:
        pred_rv = None
    
    print(f"Data period: {df.index.min()} to {df.index.max()}")
    print(f"Total days: {len(df)}")
    
    # ========================================================================
    # EQUITY STRATEGIES (Long SPY)
    # ========================================================================
    
    print("\n=== Computing Equity Strategies ===")
    
    strategies = {}
    
    # Reconstruct SPY price for trend (normalized to start at 100)
    spy_price = (1 + spy_ret).cumprod() * 100
    sma_50 = spy_price.rolling(window=50).mean()
    
    # Trend signal: Price > SMA(50), shifted to avoid lookahead
    trend_signal = (spy_price > sma_50).astype(float).shift(1).fillna(0)
    
    # 1. Buy & Hold SPY
    strategies['1. SPY Buy & Hold'] = spy_ret
    
    # 2. SMA(50) Trend Only
    strategies['2. SPY SMA(50) Trend'] = trend_signal * spy_ret
    
    # 3. Trend + VIX Sizing
    # Position: 1.0 + 0.5 * (1 - percentile_rank(VIX))
    # Low VIX → High position (1.5x)
    # High VIX → Low position (1.0x)
    vix_rank = vix.rolling(252, min_periods=50).rank(pct=True).shift(1).fillna(0.5)
    vix_position = 1.0 + 0.5 * (1.0 - vix_rank)
    strategies['3. SPY Trend + VIX Sizing'] = trend_signal * vix_position * spy_ret
    
    # 4. Trend + Prediction Sizing
    if pred_rv is not None:
        pred_rank = pred_rv.rolling(252, min_periods=50).rank(pct=True).shift(1).fillna(0.5)
        pred_position = 1.0 + 0.5 * (1.0 - pred_rank)
        strategies['4. SPY Trend + Pred Sizing'] = trend_signal * pred_position * spy_ret
    
    # ========================================================================
    # VRP STRATEGIES (Variance Swap - Sell Volatility)
    # ========================================================================
    
    print("\n=== Computing VRP Strategies ===")
    
    if pred_rv is not None:
        # Compute forward 5-day realized volatility
        rv_fwd_5 = spy_ret.rolling(5).std().shift(-5) * np.sqrt(252)
        
        # Variance swap P&L: IV^2 - RV^2 (in daily returns)
        var_swap_pnl = (vix**2 - rv_fwd_5**2).dropna()
        
        # Convert to weekly non-overlapping
        weekly_pnl = var_swap_pnl.iloc[::5]
        
        # Scale to target 10% annual volatility for fair comparison
        raw_std = weekly_pnl.std()
        target_annual_vol = 0.10
        target_weekly_std = target_annual_vol / np.sqrt(52)
        scale = target_weekly_std / raw_std
        scaled_pnl = weekly_pnl * scale
        
        # Upsample back to daily (forward fill within each week)
        daily_pnl = scaled_pnl.reindex(df.index, method='ffill').fillna(0)
        
        # 5. Unconditional (Always Sell)
        strategies['5. VRP Unconditional'] = daily_pnl
        
        # 6. VIX-Based Filtering (sell when VIX > 70th percentile)
        # IMPORTANT: Filter on WEEKLY basis, not daily!
        vix_threshold = vix.quantile(0.70)
        vix_weekly = vix.iloc[::5].reindex(scaled_pnl.index)
        vix_high_weekly = (vix_weekly > vix_threshold)
        vix_filtered_pnl_weekly = scaled_pnl.copy()
        vix_filtered_pnl_weekly[~vix_high_weekly] = 0
        # Convert to daily
        daily_pnl_vix = vix_filtered_pnl_weekly.reindex(df.index, method='ffill').fillna(0)
        strategies['6. VRP VIX-Based'] = daily_pnl_vix
        
        # 7. Residual-Based (Our Unique Alpha)
        print("Computing residual signal (expanding window)...")
        
        # VRP forecast = VIX - Pred_RV
        vrp_forecast = vix - pred_rv
        
        # Expanding window regression to compute E[VRP | VIX]
        # For efficiency, recompute every 20 days
        min_window = 252
        recalc_step = 20
        
        residuals = []
        resid_indices = []
        
        for i in range(min_window, len(df), recalc_step):
            # Historical data up to day i
            hist_vix = vix.iloc[:i].values
            hist_vrp = vrp_forecast.iloc[:i].values
            
            # Remove NaN
            valid = ~(np.isnan(hist_vix) | np.isnan(hist_vrp))
            if valid.sum() < 50:
                continue
            
            X_train = hist_vix[valid]
            y_train = hist_vrp[valid]
            
            # Fit linear model: VRP_forecast = alpha + beta * VIX
            X_mat = np.vstack([np.ones(len(X_train)), X_train]).T
            beta_coef = np.linalg.lstsq(X_mat, y_train, rcond=None)[0]
            
            # Apply to next window
            end_idx = min(i + recalc_step, len(df))
            future_vix = vix.iloc[i:end_idx].values
            future_vrp = vrp_forecast.iloc[i:end_idx].values
            
            X_future = np.vstack([np.ones(len(future_vix)), future_vix]).T
            expected_vrp = X_future @ beta_coef
            
            resid = future_vrp - expected_vrp
            residuals.extend(resid)
            resid_indices.extend(df.index[i:end_idx])
        
        residual_series = pd.Series(residuals, index=resid_indices)
        
        # Compute expanding 70th percentile of residuals
        resid_threshold = residual_series.expanding(min_periods=252).quantile(0.70).shift(1)
        
        # Signal: residual > threshold
        resid_high = (residual_series > resid_threshold).fillna(False)
        
        # Align signal with weekly P&L by forward-filling to match weekly P&L dates
        # For each weekly P&L date, use the most recent residual signal
        resid_high_weekly = pd.Series(index=scaled_pnl.index, dtype=bool)
        for date in scaled_pnl.index:
            # Find most recent residual signal on or before this date
            prior_signals = resid_high[resid_high.index <= date]
            if len(prior_signals) > 0:
                resid_high_weekly[date] = prior_signals.iloc[-1]
            else:
                resid_high_weekly[date] = False
        
        # Apply filter to WEEKLY P&L
        resid_filtered_pnl_weekly = scaled_pnl.copy()
        resid_filtered_pnl_weekly[~resid_high_weekly] = 0
        # Convert to daily
        daily_pnl_resid = resid_filtered_pnl_weekly.reindex(df.index, method='ffill').fillna(0)
        strategies['7. VRP Residual-Based'] = daily_pnl_resid
    
    # ========================================================================
    # EVALUATE ALL STRATEGIES
    # ========================================================================
    
    print("\n=== Evaluating Strategies ===")
    
    results = []
    equity_curves = {}
    
    for name, ret_series in strategies.items():
        # Drop NaN
        ret_series = ret_series.dropna()
        
        if len(ret_series) == 0:
            print(f"Skipping {name}: no valid returns")
            continue
        
        # Calculate metrics
        metrics = calculate_metrics(ret_series, benchmark=spy_ret)
        metrics['Strategy'] = name
        results.append(metrics)
        
        # Store equity curve
        equity_curves[name] = (1 + ret_series).cumprod()
    
    # Create results dataframe
    results_df = pd.DataFrame(results).set_index('Strategy')
    
    # Reorder columns for readability
    col_order = ['Total Return', 'Annualized Return', 'Annualized Vol', 
                 'Sharpe Ratio', 'Max Drawdown', 'Alpha', 'Beta', 'Correlation']
    col_order = [c for c in col_order if c in results_df.columns]
    results_df = results_df[col_order]
    
    # Display
    print("\n" + "="*80)
    print("FINAL STRATEGY RESULTS")
    print("="*80)
    print(results_df.to_string())
    print("="*80)
    
    # Save results
    results_df.to_json("artifacts/strategy/final_metrics.json", orient='index', indent=2)
    results_df.to_csv("artifacts/strategy/final_metrics.csv")
    print("\nSaved metrics to artifacts/strategy/")
    
    # ========================================================================
    # PLOTS
    # ========================================================================
    
    # Plot 1: Equity Strategies
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    equity_strats = [s for s in equity_curves.keys() if 'SPY' in s]
    for name in equity_strats:
        ax1.plot(equity_curves[name].index, equity_curves[name].values, label=name, linewidth=2)
    
    ax1.set_title("Equity Strategies (Long SPY)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Growth of $1")
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: VRP Strategies
    vrp_strats = [s for s in equity_curves.keys() if 'VRP' in s]
    for name in vrp_strats:
        ax2.plot(equity_curves[name].index, equity_curves[name].values, label=name, linewidth=2)
    
    ax2.set_title("VRP Strategies (Variance Swap)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Growth of $1")
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("artifacts/plots/strategy_comparison.png", dpi=150)
    print("\nSaved plot to artifacts/plots/strategy_comparison.png")
    
    # Plot 3: All strategies together
    plt.figure(figsize=(14, 8))
    for name, curve in equity_curves.items():
        plt.plot(curve.index, curve.values, label=name, linewidth=1.5, alpha=0.8)
    
    plt.title("All Strategies Comparison", fontsize=14, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Growth of $1 (log scale)")
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("artifacts/plots/all_strategies.png", dpi=150)
    print("Saved plot to artifacts/plots/all_strategies.png")

def main():
    cleanup_artifacts()
    run_training_pipeline()
    run_strategies_and_backtest()

if __name__ == "__main__":
    main()
