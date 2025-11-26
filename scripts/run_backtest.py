import argparse
import yaml
import pandas as pd
import numpy as np
import itertools
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from volatility_modelling.strategy.signals import load_predictions, optimize_ensemble_weights, compute_ensemble_forecast, compute_mispricing_score
from volatility_modelling.strategy.portfolio import generate_signals, calculate_weights, compute_trend_filter
from volatility_modelling.strategy.backtest import Backtester
from volatility_modelling.strategy.analytics import generate_report


def load_all_predictions(base_dir: str, horizons: list, splits: list) -> dict:
    """
    Load predictions for all 9 combinations: 3 models × 3 horizons.
    Returns dict with keys like 'garch_h2', 'lstm_rv_h5', etc.
    """
    models = ["garch", "lstm_rv", "lstm_vix"]
    all_preds = {}
    
    for h in horizons:
        preds_dir = Path(base_dir) / f"h_{h}"
        for model in models:
            model_preds = []
            for split in splits:
                path = preds_dir / f"{model}_{split}_preds.pkl"
                try:
                    data = pd.read_pickle(path)
                    pred_series = data["y_pred"]
                    model_preds.append(pred_series)
                except FileNotFoundError:
                    print(f"Warning: {path} not found.")
            
            if model_preds:
                key = f"{model}_h{h}"
                all_preds[key] = pd.concat(model_preds).sort_index()
    
    return all_preds


def run_backtest(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # Load Data
    df = pd.read_pickle(cfg["paths"]["data_path"])
    
    # Prepare Returns DataFrame for Backtester
    returns_df = pd.DataFrame(index=df.index)
    returns_df["SPY"] = df["RET_SPY"]
    
    # Construct VIX proxy returns
    vix_level = df["IV"] * 100
    returns_df["VIX"] = vix_level.pct_change()
    
    # Load train config to get horizons
    train_cfg = yaml.safe_load(open("configs/train.yaml", 'r'))
    horizons = train_cfg.get("horizons", [30])
    splits = ["train", "val", "test"]
    ret_col = train_cfg["task"]["ret_col"]
    annualization = train_cfg.get("opt", {}).get("annualization", 252)

    # Compute all forward RV columns
    for h in horizons:
        target_col = f"RV_fwd_{h}"
        if target_col not in df.columns:
            fwd_sq = 0
            for i in range(1, h + 1):
                fwd_sq = fwd_sq + df[ret_col].shift(-i).pow(2)
            df[target_col] = np.sqrt((annualization / h) * fwd_sq)

    # Load all 9 predictions (3 models × 3 horizons)
    print("=== Loading all 9 prediction streams (3 models × 3 horizons) ===")
    all_preds = load_all_predictions(cfg["paths"]["predictions_dir"], horizons, splits)
    
    if not all_preds:
        print("No predictions found. Run training first.")
        return
    
    print(f"Loaded prediction streams: {list(all_preds.keys())}")
    
    # Use a reference target for ensemble optimization
    # We'll use the middle horizon (h=5) as the reference target
    ref_horizon = horizons[len(horizons) // 2] if len(horizons) > 1 else horizons[0]
    target_col = f"RV_fwd_{ref_horizon}"
    train_end = cfg["splits"]["train_end"]
    
    # Optimize Ensemble Weights across all 9 inputs
    if cfg["ensemble"]["optimize_on_train"]:
        print(f"\nOptimizing ensemble weights on train set (target: {target_col})...")
        target = df[target_col]
        weights = optimize_ensemble_weights(all_preds, target, train_end)
        weight_dict = dict(zip(all_preds.keys(), weights))
        print("Optimized Weights:")
        for k, v in weight_dict.items():
            print(f"  {k}: {v:.4f}")
    else:
        # Equal weights
        n = len(all_preds)
        weights = [1.0 / n] * n
        weight_dict = dict(zip(all_preds.keys(), weights))

    # Compute Ensemble Forecast from all 9 inputs
    ensemble_vol = compute_ensemble_forecast(all_preds, weights)

    # Compute Mispricing Score
    score = compute_mispricing_score(ensemble_vol, df["IV"], train_end)

    # Strategy improvement flags
    spy_only_mode = cfg["strategy"].get("spy_only_mode", False)
    trend_filter_cfg = cfg["strategy"].get("trend_filter", {})
    trend_filter_enabled = trend_filter_cfg.get("enabled", False)
    
    # Compute trend filter if enabled
    trend_filter_series = None
    if trend_filter_enabled:
        ma_window = trend_filter_cfg.get("ma_window", 200)
        ma_type = trend_filter_cfg.get("ma_type", "sma")
        # Compute cumulative price from returns
        spy_price = (1 + df["RET_SPY"]).cumprod()
        trend_filter_series = compute_trend_filter(spy_price, ma_window, ma_type)
        print(f"\nTrend filter enabled: {ma_type.upper()}({ma_window})")
        print(f"  % time in uptrend: {trend_filter_series.mean()*100:.1f}%")
    
    if spy_only_mode:
        print("\nSPY-only mode enabled: VIX signals will be converted to cash")

    # Grid Search
    grid_cfg = cfg["grid_search"]
    if grid_cfg["enabled"]:
        modes = grid_cfg.get("modes", ["long_short_strict"])
        taus = grid_cfg["taus"]
        w_maxs = grid_cfg["w_maxs"]
        combinations = list(itertools.product(modes, taus, w_maxs))
    else:
        combinations = [("long_short_strict", cfg["strategy"]["params"]["tau"], cfg["strategy"]["params"]["w_max"])]

    print(f"\nRunning backtest for {len(combinations)} combinations...")

    backtester = Backtester(returns_df, cfg["strategy"]["params"]["costs_bps"])
    results = []

    # Output dir for combined ensemble - include flags in name
    output_suffix = "ensemble_9inputs"
    if spy_only_mode:
        output_suffix += "_spyonly"
    if trend_filter_enabled:
        ma_window = trend_filter_cfg.get("ma_window", 200)
        output_suffix += f"_trend{ma_window}"
    
    output_dir = Path(cfg["paths"]["output_dir"]) / output_suffix
    output_dir.mkdir(parents=True, exist_ok=True)

    for mode, tau, w_max in combinations:
        run_id = f"ensemble9_{mode}_tau_{tau}_wmax_{w_max}"

        # Update params
        params = cfg["strategy"]["params"].copy()
        params["tau"] = tau
        params["w_max"] = w_max
        params["mode"] = mode
        params["spy_only_mode"] = spy_only_mode
        params["trend_filter_enabled"] = trend_filter_enabled

        # Generate Signals with new flags
        signals = generate_signals(
            score, tau, mode,
            spy_only_mode=spy_only_mode,
            trend_filter=trend_filter_series
        )

        # Calculate Weights
        target_weights = calculate_weights(signals, ensemble_vol, params)

        # Run Backtest
        ledger = backtester.run(target_weights, cfg["strategy"]["rebalance_freq"])

        # Generate Report
        metrics = generate_report(ledger, params, str(output_dir), run_id)
        metrics["run_id"] = run_id
        results.append(metrics)

    # Summary
    summary_df = pd.DataFrame(results)
    print(f"\n=== Backtest Summary (Top 10 by Sharpe) - 9-Input Ensemble ===")
    print(summary_df.sort_values("Sharpe Ratio", ascending=False).head(10)[["run_id", "Sharpe Ratio", "Total Return", "Max Drawdown"]])

    summary_df.to_csv(output_dir / "summary.csv", index=False)
    
    # Save weights
    weights_df = pd.DataFrame([weight_dict])
    weights_df.to_csv(output_dir / "ensemble_weights.csv", index=False)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    
    run_backtest(args.cfg)
