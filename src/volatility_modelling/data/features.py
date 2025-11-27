from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

from volatility_modelling.data.loaders import DownloadSpec, download_price_series, save_series
from volatility_modelling.data.preprocessing import compute_log_returns
from volatility_modelling.utils.time import to_decimal_vol, ensure_tz_ny

@dataclass
class DataBuildConfig:
    s_and_p_proxy: str = "SPY"
    vix_ticker: str = "^VIX"
    start_date: str = "1990-01-01"
    end_date: Optional[str] = None
    horizons: List[int] = field(default_factory=lambda: [2, 5, 10, 30])
    annualization_factor: int = 252
    session: str = "America/New_York"
    output_dir: Path = Path("data")
    save_combined: bool = True


def compute_forward_realized_volatility(returns: pd.Series, horizon: int, ann_factor: int = 252) -> pd.Series:
    """Compute h-day forward realized volatility.
    
    At time t, RV_fwd_h[t] = sqrt(ann_factor) * std(r_{t+1}, ..., r_{t+h})
    
    This uses ONLY future returns (no lookahead bias in features, but target is future).
    """
    # Shift returns by -1 so rolling window captures t+1 to t+h
    future_returns = returns.shift(-1)
    rolling_std = future_returns.rolling(window=horizon, min_periods=horizon).std()
    # Shift back by h-1 so that at index t, we have std of returns t+1 to t+h
    rv = rolling_std.shift(-(horizon - 1)) * np.sqrt(ann_factor)
    rv.name = f"RV_fwd_{horizon}"
    return rv


def compute_backward_realized_volatility(returns: pd.Series, horizon: int, ann_factor: int = 252) -> pd.Series:
    """Compute h-day backward realized volatility (feature).
    
    At time t, RV_back_h[t] = sqrt(ann_factor) * std(r_{t-h+1}, ..., r_t)
    
    This uses ONLY past returns (safe to use as feature).
    """
    rv = returns.rolling(window=horizon, min_periods=horizon).std() * np.sqrt(ann_factor)
    rv.name = f"RV_back_{horizon}"
    return rv


def build_and_save_streams(cfg: DataBuildConfig) -> Dict[str, Path]:
    """Download raw data and build all required streams:

    1) S&P 500 proxy (prices + returns) for GARCH
    2) Forward realized volatility for each horizon (targets)
    3) Backward realized volatility for each horizon (features)
    4) VIX implied volatility level (IV = VIX/100) for LSTM input

    Saves: data/raw/{sp500, vix}.pkl and data/processed/timeseries.pkl

    Returns mapping of artifact names to file paths.
    """
    out = {}
    base = Path(cfg.output_dir)
    raw_dir = base / "raw"
    proc_dir = base / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    # 1) SPY prices (Adj Close) and returns
    spy_spec = DownloadSpec(ticker=cfg.s_and_p_proxy, start_date=cfg.start_date, end_date=cfg.end_date)
    spy_px = download_price_series(spy_spec)
    save_series(spy_px.rename("PX_SPY"), raw_dir / "sp500.pkl")
    out["raw_sp500"] = raw_dir / "sp500.pkl"

    spy_ret = compute_log_returns(spy_px).rename("RET_SPY")

    # 2) VIX levels -> implied vol (decimal)
    vix_spec = DownloadSpec(ticker=cfg.vix_ticker, start_date=cfg.start_date, end_date=cfg.end_date, price_field="Close")
    vix_lvl = download_price_series(vix_spec).rename("VIX")
    iv = to_decimal_vol(vix_lvl).rename("IV")
    save_series(vix_lvl, raw_dir / "vix.pkl")
    out["raw_vix"] = raw_dir / "vix.pkl"

    # 3) Compute forward and backward RV for each horizon
    rv_series = [spy_ret, iv]
    
    for h in cfg.horizons:
        rv_fwd = compute_forward_realized_volatility(spy_ret, h, cfg.annualization_factor)
        rv_back = compute_backward_realized_volatility(spy_ret, h, cfg.annualization_factor)
        rv_series.extend([rv_fwd, rv_back])
        print(f"  Computed RV_fwd_{h} and RV_back_{h}")

    # Combined table: RET_SPY, IV, RV_fwd_{h}, RV_back_{h} for all horizons
    df = pd.concat(rv_series, axis=1).dropna().sort_index()
    
    # Align combined frame to requested session timezone
    df = ensure_tz_ny(df.tz_localize(None) if df.index.tz is None else df, tz=cfg.session)
    
    if cfg.save_combined:
        df.to_pickle(proc_dir / "timeseries.pkl")
        out["combined"] = proc_dir / "timeseries.pkl"
        print(f"  Saved combined timeseries with columns: {list(df.columns)}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Total rows: {len(df)}")

    return out
