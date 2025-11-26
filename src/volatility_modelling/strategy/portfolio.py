import pandas as pd
import numpy as np


def compute_trend_filter(prices: pd.Series, ma_window: int = 200, ma_type: str = "sma") -> pd.Series:
    """
    Compute trend filter: 1 if price > MA, 0 otherwise.
    
    Args:
        prices: Price series (e.g., cumulative returns or price levels)
        ma_window: Moving average window
        ma_type: 'sma' for simple MA, 'ema' for exponential MA
    
    Returns:
        Series of 1s (uptrend) and 0s (downtrend)
    """
    if ma_type == "ema":
        ma = prices.ewm(span=ma_window, adjust=False).mean()
    else:  # sma
        ma = prices.rolling(window=ma_window).mean()
    
    return (prices > ma).astype(float)


def generate_signals(score: pd.Series, tau: float, mode: str = "long_short_strict",
                     spy_only_mode: bool = False, trend_filter: pd.Series = None) -> pd.Series:
    """
    Generate trading signals based on mispricing score S_t.
    
    Modes:
    - 'long_short_strict': 
        S <= -tau -> Long SPY (1)
        S >= +tau -> Long VIX (-1)
        Else -> Neutral (0)
        (Conservative: only trades extreme mispricing)
        
    - 'long_only_filter':
        S <= tau -> Long SPY (1)
        S > tau  -> Neutral (0)
        (Base Long SPY, exit to Cash if Model predicts much higher vol than Market)
        
    - 'long_short_aggressive':
        S <= tau -> Long SPY (1)
        S > tau  -> Long VIX (-1)
        (Always in market: SPY normally, flip to VIX if Model predicts crash)
    
    Args:
        score: Mispricing score series
        tau: Threshold for signals
        mode: Signal generation mode
        spy_only_mode: If True, convert all VIX signals (-1) to cash (0)
        trend_filter: Optional series of 1s/0s. If provided, only allow long SPY when trend=1
    """
    signals = pd.Series(0, index=score.index)
    
    if mode == "long_short_strict":
        signals[score <= -tau] = 1
        signals[score >= tau] = -1
        
    elif mode == "long_only_filter":
        # Stay long unless model predicts significantly higher vol than market
        signals[score <= tau] = 1
        signals[score > tau] = 0
        
    elif mode == "long_short_aggressive":
        # Flip to VIX only when model predicts significantly higher vol
        signals[score <= tau] = 1
        signals[score > tau] = -1
    
    # Apply SPY-only mode: convert VIX signals to cash
    if spy_only_mode:
        signals[signals == -1] = 0
    
    # Apply trend filter: only allow long SPY when in uptrend
    if trend_filter is not None:
        # Align indices
        common_idx = signals.index.intersection(trend_filter.index)
        # Where trend is down (0) and signal is long (1), go to cash (0)
        downtrend_mask = trend_filter.loc[common_idx] == 0
        signals.loc[common_idx[downtrend_mask]] = 0
        
    return signals

def calculate_weights(signals: pd.Series, ensemble_vol: pd.Series, params: dict) -> pd.DataFrame:
    """
    Calculate target weights for SPY and VIX proxy.
    SPY weight ~ 1 / V_E, capped at w_max.
    VIX weight = gamma (fixed).
    """
    w_max = params["w_max"]
    gamma = params["gamma"]
    target_vol = params.get("target_vol", 0.15) # Annualized target vol
    
    weights = pd.DataFrame(0.0, index=signals.index, columns=["SPY", "VIX"])
    
    # SPY Sizing: Volatility Targeting
    # w_spy = target_vol / forecast_vol
    # We use ensemble_vol as the forecast
    # Cap at w_max
    
    raw_spy_weight = target_vol / ensemble_vol
    capped_spy_weight = raw_spy_weight.clip(upper=w_max)
    
    # Apply signals
    # Signal 1: Long SPY
    weights.loc[signals == 1, "SPY"] = capped_spy_weight.loc[signals == 1]
    
    # Signal -1: Long VIX
    weights.loc[signals == -1, "VIX"] = gamma
    
    # Signal 0: Neutral (Cash) - weights remain 0
    
    return weights
