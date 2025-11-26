import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Tuple

def load_predictions(predictions_dir: str, splits: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Load predictions for GARCH, LSTM_RV, LSTM_VIX for all splits.
    Returns a dictionary of DataFrames aligned by index.
    """
    models = ["garch", "lstm_rv", "lstm_vix"]
    preds = {}
    
    for model in models:
        model_preds = []
        for split in splits:
            path = f"{predictions_dir}/{model}_{split}_preds.pkl"
            try:
                df = pd.read_pickle(path)
                # df is a dict with y_true, y_pred
                # We want a Series of y_pred indexed by date
                pred_series = df["y_pred"]
                if isinstance(pred_series, np.ndarray):
                    # If it's an array, we need the index from y_true or the original df
                    # The pickle saved in train_loop has y_true as Series (hopefully) or we need to reconstruct
                    # In train_loop: 
                    # predictions[split_name] = { "y_true": y_true, "y_pred": y_pred }
                    # y_true was a Series, y_pred was a Series (aligned).
                    pass
                model_preds.append(pred_series)
            except FileNotFoundError:
                print(f"Warning: {path} not found.")
                
        if model_preds:
            preds[model] = pd.concat(model_preds).sort_index()
            
    return preds

def optimize_ensemble_weights(preds: Dict[str, pd.Series], target: pd.Series, train_end: str) -> List[float]:
    """
    Find convex combination weights that minimize RMSE on the training set.
    """
    # Align all series
    df = pd.DataFrame(preds)
    df["target"] = target
    
    train_df = df.loc[:train_end].dropna()
    
    X = train_df[list(preds.keys())].values
    y = train_df["target"].values
    
    def objective(weights):
        y_pred = np.dot(X, weights)
        return np.sqrt(np.mean((y - y_pred)**2))
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(len(preds))]
    initial_weights = [1.0/len(preds)] * len(preds)
    
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def compute_ensemble_forecast(preds: Dict[str, pd.Series], weights: List[float]) -> pd.Series:
    """
    Compute weighted average of predictions.
    """
    df = pd.DataFrame(preds)
    return df.dot(weights)

def compute_mispricing_score(ensemble_vol: pd.Series, iv: pd.Series, train_end: str) -> pd.Series:
    """
    Compute score S_t = (V_E - IV) / sigma_S
    sigma_S is std of (V_E - IV) on train set.
    """
    diff = ensemble_vol - iv
    
    # Calculate std on train set only
    train_diff = diff.loc[:train_end]
    sigma_s = train_diff.std()
    
    return diff / sigma_s
