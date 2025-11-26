import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

def calculate_metrics(returns: pd.Series, freq: int = 252) -> dict:
    """
    Calculate standard performance metrics.
    """
    if len(returns) == 0:
        return {}
        
    total_return = (1 + returns).prod() - 1
    cagr = (1 + total_return) ** (freq / len(returns)) - 1
    vol = returns.std() * np.sqrt(freq)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(freq) if vol > 0 else 0
    
    downside_returns = returns[returns < 0]
    sortino = (returns.mean() / downside_returns.std()) * np.sqrt(freq) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
    
    # Max Drawdown
    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()
    
    return {
        "Total Return": float(total_return),
        "CAGR": float(cagr),
        "Volatility": float(vol),
        "Sharpe Ratio": float(sharpe),
        "Sortino Ratio": float(sortino),
        "Max Drawdown": float(max_dd)
    }

def generate_report(ledger: pd.DataFrame, params: dict, output_dir: str, run_id: str):
    """
    Save metrics, plots, and ledger.
    """
    out_path = Path(output_dir) / run_id
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Metrics
    metrics = calculate_metrics(ledger["Net_Ret"])
    metrics["params"] = params
    
    with open(out_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Ledger
    ledger.to_csv(out_path / "ledger.csv")
    
    # Plots
    plt.figure(figsize=(12, 8))
    
    # Equity Curve
    plt.subplot(2, 1, 1)
    ledger["Equity"].plot(title="Equity Curve")
    plt.grid(True, alpha=0.3)
    
    # Weights
    plt.subplot(2, 1, 2)
    ledger[["SPY", "VIX"]].plot(ax=plt.gca(), title="Position Weights", alpha=0.7)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path / "performance.png")
    plt.close()
    
    return metrics
