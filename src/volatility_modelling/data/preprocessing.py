import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

class Preprocessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.log_cols = cfg.get("log_transform_cols", [])

    def check_stationarity(self, df, name="Dataset"):
        print(f"\n--- Stationarity Checks ({name}) ---")
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                clean_series = df[col].dropna()
                # ADF test requires some variance
                if len(clean_series) > 20 and clean_series.std() > 0:
                    try:
                        result = adfuller(clean_series)
                        print(f"  {col}: p-value={result[1]:.4f} ({'Stationary' if result[1]<0.05 else 'Non-Stationary'})")
                    except Exception as e:
                        print(f"  {col}: ADF Test Failed ({e})")

    def fit(self, train_df):
        """
        Fit preprocessing on training data (placeholder for future extensions).
        """
        return self

    def transform(self, df):
        """
        Apply transformations.
        """
        df = df.copy()
                
        # Log Transform
        for col in self.log_cols:
            if col in df.columns:
                # Ensure positive before log: replace non-positive values with a small epsilon
                if (df[col] <= 0).any():
                    # compute a sensible epsilon: 10% of the smallest positive value in the column
                    positive_vals = df.loc[df[col] > 0, col]
                    if len(positive_vals) > 0:
                        eps = float(positive_vals.min()) * 0.1
                        eps = max(eps, 1e-12)
                    else:
                        eps = 1e-8
                    print(f"  [Preprocessing] Warning: {col} has non-positive values. Replacing with eps={eps:.2e} before log-transform.")
                    df.loc[df[col] <= 0, col] = eps
                # additionally clip extremely small values to avoid -inf after log
                df[col] = df[col].clip(lower=1e-12)
                df[col] = np.log(df[col])
                
        return df

    def inverse_transform_target(self, y_pred, target_col):
        """
        Inverse transform the target variable if it was transformed.
        """
        if target_col in self.log_cols:
            return np.exp(y_pred)
        return y_pred

def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute logarithmic returns from a price series."""
    return np.log(prices / prices.shift(1)).dropna()
