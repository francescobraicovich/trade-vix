from __future__ import annotations

import pandas as pd
import numpy as np

def realized_volatility_forward_window(
    returns: pd.Series, window: int = 30, ann_factor: int = 252
) -> pd.Series:
    """Compute *forward* realized volatility over the next `window` trading days.


    At each date t, RV_{t+window} = sqrt(ann_factor) * sqrt( (1/window) * sum_{i=1..window} (r_{t+i} - mean_{t+1..t+window})^2 ).

    The result is indexed by the forecast origin date t.

    """
    r = returns.sort_index()
    # forward window using shift(-1)
    fwd = r.shift(-1).rolling(window=window, min_periods=window)
    mu = fwd.mean()
    var = (fwd.apply(lambda x: ((x - x.mean()) ** 2).sum(), raw=False) / float(window))
    rv = np.sqrt(var) * np.sqrt(ann_factor)
    rv.name = f"RV{window}"
    return rv

def align_target_with_features(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Join features and target on the intersection of dates, dropping any rows with NA."""
    df = features.copy()
    df[target.name] = target
    return df.dropna()
