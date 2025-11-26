from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

import pandas as pd

import yfinance as yf

try:  # Disable yfinance caches when sqlite is missing; keeps downloads working on minimal installs.
    import sqlite3  # noqa: F401
except Exception:  # pragma: no cover - depends on system setup
    from yfinance import cache as _yf_cache

    _yf_cache._TzCacheManager._tz_cache = _yf_cache._TzCacheDummy()
    _yf_cache._CookieCacheManager._Cookie_cache = _yf_cache._CookieCacheDummy()
    _yf_cache._ISINCacheManager._isin_cache = _yf_cache._ISINCacheDummy()

from volatility_modelling.utils.time import parse_date, coerce_end_date, ensure_tz_ny

PriceField = Literal["Adj Close", "Close"]

@dataclass
class DownloadSpec:
    ticker: str
    start_date: str
    end_date: Optional[str] = None
    price_field: PriceField = "Adj Close"

def _fetch_yf_series(ticker: str, start: str, end: Optional[str], price_field: PriceField = "Adj Close") -> pd.Series:
    start_ts = parse_date(start)
    end_ts = coerce_end_date(parse_date(end))
    df = yf.download(
        ticker,
        start=start_ts.tz_convert(None),
        end=end_ts.tz_convert(None),
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker} between {start} and {end}.")

    if isinstance(df.columns, pd.MultiIndex):
        for label in ((price_field, ticker), (ticker, price_field)):
            if label in df.columns:
                series = df[label]
                break
        else:
            raise RuntimeError(
                f"Expected column '{price_field}' for {ticker} not found in Yahoo Finance frame: {df.columns}"
            )
    else:
        if price_field not in df.columns:
            raise RuntimeError(f"Expected column '{price_field}' not found in Yahoo Finance frame: {df.columns}")
        series = df[price_field]

    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    series = series.copy()
    series.name = ticker
    series.index = pd.to_datetime(series.index)
    series = ensure_tz_ny(series)
    return series.sort_index()


def download_price_series(spec: DownloadSpec) -> pd.Series:
    return _fetch_yf_series(spec.ticker, spec.start_date, spec.end_date, spec.price_field)

def save_series(s: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Persist using pandas native pickle format for simplicity
    s.to_pickle(path)

def load_series(path: Path) -> pd.Series:
    return pd.read_pickle(path)
