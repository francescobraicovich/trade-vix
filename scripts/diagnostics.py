import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
import numpy as np

def plot_diagnostics(y_true, y_pred, title, save_dir, prefix):
    # Scatter Plot
    plt.figure(figsize=(8, 8))
    sns.set_style("whitegrid")
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    
    # Identity line
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  # min of both axes
        np.max([plt.xlim(), plt.ylim()]),  # max of both axes
    ]
    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Actual Volatility")
    plt.ylabel("Predicted Volatility")
    plt.title(f"{title} - Scatter")
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_scatter.png", dpi=150)
    plt.close()

    # Residual Plot
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 6))
    plt.scatter(y_true.index, residuals, alpha=0.5, s=10)
    plt.axhline(0, color='k', linestyle='--', alpha=0.7)
    plt.xlabel("Date")
    plt.ylabel("Residuals (True - Pred)")
    plt.title(f"{title} - Residuals")
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_residuals.png", dpi=150)
    plt.close()
    
    # Residual Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=50)
    plt.title(f"{title} - Residual Distribution")
    plt.xlabel("Residual")
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_res_dist.png", dpi=150)
    plt.close()

def run_diagnostics(train_cfg_path):
    with open(train_cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    horizons = cfg.get("horizons", [30])
    artifacts_base = Path(cfg["paths"]["artifacts_dir"])
    models = ["garch", "lstm_rv", "lstm_vix"]
    splits = ["val", "test"] # Train is usually less interesting for diagnostics, but can add if needed

    for h in horizons:
        print(f"Processing horizon {h}...")
        artifacts_dir = artifacts_base / f"h_{h}"
        diag_dir = artifacts_dir / "diagnostics"
        diag_dir.mkdir(exist_ok=True)
        
        for model in models:
            for split in splits:
                pred_path = artifacts_dir / f"{model}_{split}_preds.pkl"
                if not pred_path.exists():
                    print(f"  Warning: {pred_path} not found. Skipping.")
                    continue
                
                data = pd.read_pickle(pred_path)
                # data is dict {'y_true': Series, 'y_pred': Series}
                y_true = data["y_true"]
                y_pred = data["y_pred"]
                
                # Ensure alignment
                common_idx = y_true.index.intersection(y_pred.index)
                y_true = y_true.loc[common_idx]
                y_pred = y_pred.loc[common_idx]
                
                if len(y_true) == 0:
                    print(f"  Warning: No overlapping data for {model} {split} h={h}.")
                    continue
                
                title = f"{model.upper()} h={h} {split.capitalize()}"
                prefix = f"{model}_{split}"
                
                plot_diagnostics(y_true, y_pred, title, diag_dir, prefix)
                print(f"  Generated plots for {title}")

if __name__ == "__main__":
    run_diagnostics("configs/train.yaml")
