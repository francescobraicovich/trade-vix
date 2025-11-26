from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

import pandas as pd

from volatility_modelling.data.loaders import DownloadSpec, download_price_series, save_series
from volatility_modelling.data.preprocessing import compute_log_returns, clip_outliers
from volatility_modelling.data.target import realized_volatility_forward_window
from volatility_modelling.utils.time import to_decimal_vol, ensure_tz_ny

@dataclass
class DataBuildConfig:
    s_and_p_proxy: str = "SPY"
    vix_ticker: str = "^VIX"
    start_date: str = "1990-01-01"
    end_date: Optional[str] = None
    realized_vol_window: int = 30
    annualization_factor: int = 252
    session: str = "America/New_York"
    output_dir: Path = Path("data")
    save_combined: bool = True

def build_and_save_streams(cfg: DataBuildConfig) -> Dict[str, Path]:
    """Download raw data and build the three required streams:

    1) S&P 500 proxy (prices + returns) for GARCH

    2) Realized 30-day volatility from SPY returns (RV) for LSTM target

    3) VIX implied volatility level (IV = VIX/100) for LSTM input

    Saves: data/raw/{SPY, VIX}.pkl and data/processed/{rv, combined}.pkl

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

    spy_ret = clip_outliers(compute_log_returns(spy_px)).rename("RET_SPY")

    # 2) Realized volatility over forward 30-day window
    rv = realized_volatility_forward_window(spy_ret, window=cfg.realized_vol_window, ann_factor=cfg.annualization_factor)
    save_series(rv.rename("RV30"), proc_dir / "rv30.pkl")
    out["rv30"] = proc_dir / "rv30.pkl"

    # 3) VIX levels -> implied vol (decimal)
    vix_spec = DownloadSpec(ticker=cfg.vix_ticker, start_date=cfg.start_date, end_date=cfg.end_date, price_field="Close")
    vix_lvl = download_price_series(vix_spec).rename("VIX")
    iv = to_decimal_vol(vix_lvl).rename("IV")
    save_series(vix_lvl, raw_dir / "vix.pkl")
    out["raw_vix"] = raw_dir / "vix.pkl"

    # Combined table (optional): RET_SPY, RV30, IV
    df = pd.concat([spy_ret, rv.rename("RV30"), iv.rename("IV")], axis=1).dropna().sort_index()
    # Align combined frame to requested session timezone
    df = ensure_tz_ny(df.tz_localize(None) if df.index.tz is None else df, tz=cfg.session)
    if cfg.save_combined:
        df.to_pickle(proc_dir / "timeseries.pkl")
        out["combined"] = proc_dir / "timeseries.pkl"

    return out
