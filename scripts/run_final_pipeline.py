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

def run_download_data():
    python_exec = sys.executable
    run_command(
        f"{python_exec} scripts/download_data.py",
        "Downloading Latest Data"
    )

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

def run_garch_diagnostics():
    python_exec = sys.executable
    run_command(
        f"{python_exec} scripts/garch_diagnostics.py",
        "GARCH Grid Search & Diagnostics"
    )

def generate_data_plots():
    python_exec = sys.executable
    run_command(
        f"{python_exec} scripts/generate_data_plots.py",
        "Generating Data Plots"
    )

def run_training_pipeline():
    python_exec = sys.executable
    
    # Train Models
    # We need to train for each horizon? 
    # run_train.py handles multiple horizons if specified in config.
    # Let's check train.yaml to see if the horizons are set there.
    
    # Train GARCH
    run_command(
        f"{python_exec} scripts/run_train.py --train_cfg configs/train.yaml --model_cfg configs/model/garch.yaml",
        "Training GARCH"
    )
    
    # Train LSTM-RV
    run_command(
        f"{python_exec} scripts/run_train.py --train_cfg configs/train.yaml --model_cfg configs/model/lstm_rv.yaml",
        "Training LSTM-RV"
    )
    
    # Train LSTM-VIX
    run_command(
        f"{python_exec} scripts/run_train.py --train_cfg configs/train.yaml --model_cfg configs/model/lstm_vix.yaml",
        "Training LSTM-VIX"
    )

def load_predictions(horizons=[2, 5, 10, 30]):
    """Load predictions from all models and horizons, separated by split."""
    base_dir = Path("artifacts")
    models = ['garch', 'lstm_rv', 'lstm_vix']
    splits = ['train', 'val', 'test']
    
    predictions = {split: {} for split in splits}
    
    for h in horizons:
        for model in models:
            for split in splits:
                p = base_dir / f"h_{h}" / f"{model}_{split}_preds.pkl"
                if p.exists():
                    data = pd.read_pickle(p)
                    if isinstance(data, dict):
                        pred_series = data['y_pred']
                    else:
                        pred_series = data
                    
                    predictions[split][f"{model}_h{h}"] = pred_series
    
    # Convert to DataFrames per split
    result = {}
    for split in splits:
        if predictions[split]:
            result[split] = pd.DataFrame(predictions[split])
        else:
            result[split] = pd.DataFrame()
    
    return result

def optimize_ensemble_weights(preds_val, models, horizon, num_samples=20):
    """
    Grid search for optimal ensemble weights on validation set.
    Returns best weights based on minimum RMSE.
    """
    from sklearn.metrics import mean_squared_error
    from itertools import product
    
    # Generate weight combinations that sum to 1
    # Using a coarse grid for efficiency
    step = 1.0 / (num_samples - 1) if num_samples > 1 else 1.0
    weight_options = [i * step for i in range(num_samples)]
    
    model_cols = [f"{m}_h{horizon}" for m in models]
    
    # Check if all model predictions exist
    available_cols = [col for col in model_cols if col in preds_val.columns]
    if len(available_cols) < len(models):
        print(f"Warning: Missing predictions for horizon {horizon}. Available: {available_cols}")
        # Return equal weights for available models
        n = len(available_cols)
        return {col.split('_h')[0]: 1.0/n for col in available_cols}
    
    # Get actual RV values from the data
    df = pd.read_pickle("data/processed/timeseries.pkl")
    rv_col = f"RV_fwd_{horizon}"
    
    # Align predictions with actual values
    common_idx = preds_val.index.intersection(df.index)
    y_true = df.loc[common_idx, rv_col].dropna()
    
    best_weights = None
    best_rmse = float('inf')
    
    print(f"\nOptimizing ensemble weights for horizon {horizon}...")
    print(f"Validation samples: {len(y_true)}")
    
    # Generate all weight combinations
    for weights in product(weight_options, repeat=len(models)):
        # Skip if weights don't sum to 1 (with tolerance)
        if not np.isclose(sum(weights), 1.0, atol=0.01):
            continue
        
        # Compute ensemble prediction (geometric mean in log space)
        ensemble_pred = None
        for i, col in enumerate(model_cols):
            pred = preds_val.loc[common_idx, col]
            # Align with y_true
            pred = pred.reindex(y_true.index)
            
            log_pred = np.log(pred)
            if ensemble_pred is None:
                ensemble_pred = weights[i] * log_pred
            else:
                ensemble_pred += weights[i] * log_pred
        
        ensemble_pred = np.exp(ensemble_pred)
        
        # Calculate RMSE
        valid_mask = ~np.isnan(ensemble_pred) & ~np.isnan(y_true)
        if valid_mask.sum() < 10:
            continue
        
        rmse = np.sqrt(mean_squared_error(y_true[valid_mask], ensemble_pred[valid_mask]))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = {models[i]: weights[i] for i in range(len(models))}
    
    if best_weights is None:
        # Fallback to equal weights
        best_weights = {m: 1.0/len(models) for m in models}
        print(f"Warning: Could not optimize weights. Using equal weights.")
    else:
        print(f"Best weights: {best_weights}")
        print(f"Validation RMSE: {best_rmse:.6f}")
    
    return best_weights

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
    Implement all strategies from strategies.md across multiple horizons.
    
    For each horizon (2, 5, 10, 30 days):
    1. Optimize ensemble weights on validation set
    2. Test strategies on test set
    
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
    print("\n=== Running Backtests Across All Horizons ===")
    
    # Load Data
    df = pd.read_pickle("data/processed/timeseries.pkl")
    
    # Load Predictions (separated by split)
    preds_by_split = load_predictions(horizons=[2, 5, 10, 30])
    
    if all(p.empty for p in preds_by_split.values()):
        print("ERROR: No predictions found! Run training first.")
        return
    
    # Get train/val/test splits
    from volatility_modelling.data.splits import split_by_date, DateSplits
    import yaml
    
    with open("configs/train.yaml", 'r') as f:
        train_cfg = yaml.safe_load(f)
    
    splits_cfg = train_cfg["splits"]
    splits = DateSplits(
        train_end=splits_cfg["train_end"],
        valid_end=splits_cfg["valid_end"],
        test_end=splits_cfg["test_end"]
    )
    
    train_df, val_df, test_df = split_by_date(df, splits)
    
    print(f"Data splits:")
    print(f"  Train: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} days)")
    print(f"  Val:   {val_df.index.min()} to {val_df.index.max()} ({len(val_df)} days)")
    print(f"  Test:  {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} days)")
    
    # Store all results
    all_results = []
    all_equity_curves = {}
    ensemble_weights = {}
    
    models = ['garch', 'lstm_rv', 'lstm_vix']
    horizons = [2, 5, 10, 30]
    
    # Optimize ensemble weights for each horizon on validation set
    print("\n" + "="*80)
    print("STEP 1: OPTIMIZING ENSEMBLE WEIGHTS ON VALIDATION SET")
    print("="*80)
    
    for h in horizons:
        ensemble_weights[h] = optimize_ensemble_weights(
            preds_by_split['val'], 
            models, 
            h, 
            num_samples=11  # 0.0, 0.1, 0.2, ..., 1.0
        )
    
    # Save ensemble weights
    ensemble_weights_df = pd.DataFrame(ensemble_weights).T
    ensemble_weights_df.to_csv("artifacts/strategy/ensemble_weights.csv")
    print("\nSaved ensemble weights to artifacts/strategy/ensemble_weights.csv")
    
    # ========================================================================
    # STEP 2: BACKTEST ALL STRATEGIES ON TEST SET FOR EACH HORIZON
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 2: BACKTESTING STRATEGIES ON TEST SET")
    print("="*80)
    
    # Use test set for final evaluation
    df_test = test_df
    spy_ret = df_test['RET_SPY']
    vix = df_test['IV']
    
    # Get test predictions
    preds_test = preds_by_split['test']
    
    # Align
    common_idx = df_test.index.intersection(preds_test.index) if not preds_test.empty else df_test.index
    df_test = df_test.loc[common_idx]
    spy_ret = spy_ret.loc[common_idx]
    vix = vix.loc[common_idx]
    if not preds_test.empty:
        preds_test = preds_test.loc[common_idx]
    
    print(f"\nTest period: {df_test.index.min()} to {df_test.index.max()}")
    print(f"Test days: {len(df_test)}")
    
    # Common trend signal (same for all horizons)
    spy_price = (1 + spy_ret).cumprod() * 100
    sma_50 = spy_price.rolling(window=50).mean()
    trend_signal = (spy_price > sma_50).astype(float).shift(1).fillna(0)
    
    # VIX-based position sizing (same for all horizons)
    vix_rank = vix.rolling(252, min_periods=50).rank(pct=True).shift(1).fillna(0.5)
    vix_position = 1.0 + 0.5 * (1.0 - vix_rank)
    
    # ========================================================================
    # LOOP OVER HORIZONS
    # ========================================================================
    
    for h in horizons:
        print(f"\n{'='*80}")
        print(f"HORIZON: {h} days")
        print(f"{'='*80}")
        
        strategies = {}
        
        # Build ensemble prediction using optimized weights
        weights = ensemble_weights[h]
        print(f"Using ensemble weights: {weights}")
        
        ensemble_pred = None
        for model in models:
            col = f"{model}_h{h}"
            if col in preds_test.columns and model in weights:
                pred = preds_test[col]
                log_pred = np.log(pred)
                if ensemble_pred is None:
                    ensemble_pred = weights[model] * log_pred
                else:
                    ensemble_pred += weights[model] * log_pred
        
        if ensemble_pred is not None:
            pred_rv = np.exp(ensemble_pred)
        else:
            pred_rv = None
            print(f"Warning: Could not build ensemble for horizon {h}")
        
        # ====================================================================
        # EQUITY STRATEGIES (Long SPY)
        # ====================================================================
        
        # 1. Buy & Hold SPY (same for all horizons)
        strategies[f'1. SPY Buy & Hold (h={h})'] = spy_ret
        
        # 2. SMA(50) Trend Only (same for all horizons)
        strategies[f'2. SPY SMA(50) Trend (h={h})'] = trend_signal * spy_ret
        
        # 3. Trend + VIX Sizing (same for all horizons)
        strategies[f'3. SPY Trend + VIX Sizing (h={h})'] = trend_signal * vix_position * spy_ret
        
        # 4. Trend + Prediction Sizing (horizon-specific)
        if pred_rv is not None:
            pred_rank = pred_rv.rolling(252, min_periods=50).rank(pct=True).shift(1).fillna(0.5)
            pred_position = 1.0 + 0.5 * (1.0 - pred_rank)
            strategies[f'4. SPY Trend + Pred Sizing (h={h})'] = trend_signal * pred_position * spy_ret
        
        # ====================================================================
        # VRP STRATEGIES (Variance Swap - Sell Volatility)
        # ====================================================================
        
        if pred_rv is not None:
            # Use actual forward RV from the data for variance swap P&L
            rv_fwd_col = f"RV_fwd_{h}"
            if rv_fwd_col in df_test.columns:
                rv_fwd = df_test[rv_fwd_col]
            else:
                # Fallback: compute from returns
                rv_fwd = spy_ret.rolling(h).std().shift(-h) * np.sqrt(252)
            
            # Variance swap P&L: IV^2 - RV^2
            var_swap_pnl = (vix**2 - rv_fwd**2).dropna()
            
            # Convert to non-overlapping (every h days)
            non_overlap_pnl = var_swap_pnl.iloc[::h]
            
            # Scale to target 10% annual volatility for fair comparison
            raw_std = non_overlap_pnl.std()
            target_annual_vol = 0.10
            periods_per_year = 252 / h
            target_period_std = target_annual_vol / np.sqrt(periods_per_year)
            scale = target_period_std / raw_std if raw_std > 0 else 1.0
            scaled_pnl = non_overlap_pnl * scale
            
            # Upsample back to daily (forward fill within each period)
            daily_pnl = scaled_pnl.reindex(df_test.index, method='ffill').fillna(0)
            
            # 5. Unconditional (Always Sell)
            strategies[f'5. VRP Unconditional (h={h})'] = daily_pnl
            
            # 6. VIX-Based Filtering (sell when VIX > 70th percentile)
            vix_threshold = vix.quantile(0.70)
            vix_periodic = vix.iloc[::h].reindex(scaled_pnl.index)
            vix_high_periodic = (vix_periodic > vix_threshold)
            vix_filtered_pnl_periodic = scaled_pnl.copy()
            vix_filtered_pnl_periodic[~vix_high_periodic] = 0
            # Convert to daily
            daily_pnl_vix = vix_filtered_pnl_periodic.reindex(df_test.index, method='ffill').fillna(0)
            strategies[f'6. VRP VIX-Based (h={h})'] = daily_pnl_vix
            
            # 7. Residual-Based (Our Unique Alpha)
            # VRP forecast = VIX - Pred_RV
            vrp_forecast = vix - pred_rv
            
            # Expanding window regression to compute E[VRP | VIX]
            min_window = 252
            recalc_step = 20
            
            residuals = []
            resid_indices = []
            
            for i in range(min_window, len(df_test), recalc_step):
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
                end_idx = min(i + recalc_step, len(df_test))
                future_vix = vix.iloc[i:end_idx].values
                future_vrp = vrp_forecast.iloc[i:end_idx].values
                
                X_future = np.vstack([np.ones(len(future_vix)), future_vix]).T
                expected_vrp = X_future @ beta_coef
                
                resid = future_vrp - expected_vrp
                residuals.extend(resid)
                resid_indices.extend(df_test.index[i:end_idx])
            
            if len(residuals) > 0:
                residual_series = pd.Series(residuals, index=resid_indices)
                
                # Compute expanding 70th percentile of residuals
                resid_threshold = residual_series.expanding(min_periods=252).quantile(0.70).shift(1)
                
                # Signal: residual > threshold
                resid_high = (residual_series > resid_threshold).fillna(False)
                
                # Align signal with periodic P&L
                resid_high_periodic = pd.Series(index=scaled_pnl.index, dtype=bool)
                for date in scaled_pnl.index:
                    prior_signals = resid_high[resid_high.index <= date]
                    if len(prior_signals) > 0:
                        resid_high_periodic[date] = prior_signals.iloc[-1]
                    else:
                        resid_high_periodic[date] = False
                
                # Apply filter to periodic P&L
                resid_filtered_pnl_periodic = scaled_pnl.copy()
                resid_filtered_pnl_periodic[~resid_high_periodic] = 0
                # Convert to daily
                daily_pnl_resid = resid_filtered_pnl_periodic.reindex(df_test.index, method='ffill').fillna(0)
                strategies[f'7. VRP Residual-Based (h={h})'] = daily_pnl_resid
        
        # ====================================================================
        # EVALUATE STRATEGIES FOR THIS HORIZON
        # ====================================================================
        
        for name, ret_series in strategies.items():
            ret_series = ret_series.dropna()
            
            if len(ret_series) == 0:
                print(f"Skipping {name}: no valid returns")
                continue
            
            # Calculate metrics
            metrics = calculate_metrics(ret_series, benchmark=spy_ret)
            metrics['Strategy'] = name
            metrics['Horizon'] = h
            all_results.append(metrics)
            
            # Store equity curve
            all_equity_curves[name] = (1 + ret_series).cumprod()
    
    # ========================================================================
    # STEP 3: COMPILE AND SAVE RESULTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 3: FINAL RESULTS")
    print("="*80)
    
    # Create results dataframe
    results_df = pd.DataFrame(all_results)
    
    # Reorder columns
    col_order = ['Strategy', 'Horizon', 'Total Return', 'Annualized Return', 
                 'Annualized Vol', 'Sharpe Ratio', 'Max Drawdown', 
                 'Alpha', 'Beta', 'Correlation']
    col_order = [c for c in col_order if c in results_df.columns]
    results_df = results_df[col_order]
    
    # Display summary
    print("\n" + results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv("artifacts/strategy/final_metrics_all_horizons.csv", index=False)
    results_df.to_json("artifacts/strategy/final_metrics_all_horizons.json", orient='records', indent=2)
    print("\n✓ Saved detailed results to artifacts/strategy/final_metrics_all_horizons.*")
    
    # Create summary by strategy type
    results_df['Strategy_Type'] = results_df['Strategy'].apply(
        lambda x: 'SPY' if 'SPY' in x else 'VRP'
    )
    
    # Best strategy per horizon
    print("\n" + "="*80)
    print("BEST STRATEGIES BY HORIZON (Sorted by Sharpe Ratio)")
    print("="*80)
    for h in horizons:
        h_results = results_df[results_df['Horizon'] == h].sort_values('Sharpe Ratio', ascending=False)
        print(f"\nHorizon {h} days:")
        print(h_results[['Strategy', 'Sharpe Ratio', 'Annualized Return', 'Max Drawdown']].head(3).to_string(index=False))
    
    # ========================================================================
    # PLOTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    # Plot 1: Equity Strategies by Horizon
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, h in enumerate(horizons):
        ax = axes[idx]
        equity_strats = [s for s in all_equity_curves.keys() if 'SPY' in s and f'(h={h})' in s]
        
        for name in equity_strats:
            label = name.split(' (h=')[0]  # Remove horizon from label
            ax.plot(all_equity_curves[name].index, all_equity_curves[name].values, 
                   label=label, linewidth=2, alpha=0.8)
        
        ax.set_title(f"Equity Strategies - Horizon {h} days", fontsize=12, fontweight='bold')
        ax.set_ylabel("Growth of $1")
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig("artifacts/plots/equity_strategies_by_horizon.png", dpi=150)
    print("✓ Saved artifacts/plots/equity_strategies_by_horizon.png")
    plt.close()
    
    # Plot 2: VRP Strategies by Horizon
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, h in enumerate(horizons):
        ax = axes[idx]
        vrp_strats = [s for s in all_equity_curves.keys() if 'VRP' in s and f'(h={h})' in s]
        
        for name in vrp_strats:
            label = name.split(' (h=')[0]  # Remove horizon from label
            ax.plot(all_equity_curves[name].index, all_equity_curves[name].values, 
                   label=label, linewidth=2, alpha=0.8)
        
        ax.set_title(f"VRP Strategies - Horizon {h} days", fontsize=12, fontweight='bold')
        ax.set_ylabel("Growth of $1")
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("artifacts/plots/vrp_strategies_by_horizon.png", dpi=150)
    print("✓ Saved artifacts/plots/vrp_strategies_by_horizon.png")
    plt.close()
    
    # Plot 3: Sharpe Ratio Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # SPY strategies
    spy_results = results_df[results_df['Strategy_Type'] == 'SPY']
    for strat in spy_results['Strategy'].str.split(' \\(h=').str[0].unique():
        strat_data = spy_results[spy_results['Strategy'].str.contains(strat.replace('(', '\\(').replace(')', '\\)'))]
        ax1.plot(strat_data['Horizon'], strat_data['Sharpe Ratio'], 
                marker='o', linewidth=2, label=strat)
    
    ax1.set_title("Equity Strategies - Sharpe Ratio vs Horizon", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Horizon (days)")
    ax1.set_ylabel("Sharpe Ratio")
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(horizons)
    
    # VRP strategies
    vrp_results = results_df[results_df['Strategy_Type'] == 'VRP']
    for strat in vrp_results['Strategy'].str.split(' \\(h=').str[0].unique():
        strat_data = vrp_results[vrp_results['Strategy'].str.contains(strat.replace('(', '\\(').replace(')', '\\)'))]
        ax2.plot(strat_data['Horizon'], strat_data['Sharpe Ratio'], 
                marker='o', linewidth=2, label=strat)
    
    ax2.set_title("VRP Strategies - Sharpe Ratio vs Horizon", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Horizon (days)")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(horizons)
    
    plt.tight_layout()
    plt.savefig("artifacts/plots/sharpe_ratio_comparison.png", dpi=150)
    print("✓ Saved artifacts/plots/sharpe_ratio_comparison.png")
    plt.close()
    
    print("\n" + "="*80)
    print("BACKTESTING COMPLETE")
    print("="*80)

def main():
    cleanup_artifacts()
    run_download_data()
    generate_data_plots()
    run_garch_diagnostics()
    run_training_pipeline()
    run_strategies_and_backtest()

if __name__ == "__main__":
    main()
