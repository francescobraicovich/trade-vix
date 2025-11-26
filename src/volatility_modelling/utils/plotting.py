import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_predictions(y_true: pd.Series, y_pred: pd.Series, title: str, save_path: str):
    """
    Plots true vs predicted values.
    """
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    plt.plot(y_true.index, y_true.values, label="Actual", alpha=0.7, linewidth=1.5)
    plt.plot(y_pred.index, y_pred.values, label="Predicted", alpha=0.8, linewidth=1.5, linestyle='--')
    
    plt.title(title, fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Volatility", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
