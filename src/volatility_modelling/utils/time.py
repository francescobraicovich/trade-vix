from __future__ import annotations

import datetime as dt
from typing import Optional

import pandas as pd
from pandas.tseries.offsets import BDay

NY_TZ = "America/New_York"

def today_ny() -> pd.Timestamp:
    """Return today's date in New York timezone (normalized to date without time)."""
    now = pd.Timestamp.now(tz=NY_TZ)
    return pd.Timestamp(now.date(), tz=NY_TZ)

def parse_date(s: Optional[str]) -> Optional[pd.Timestamp]:
    """Parse an ISO-like date string to a timezone-aware Timestamp in NY time.


    If s is None, returns None.

    """
    if s is None:
        return None
    ts = pd.Timestamp(s)
    if ts.tz is None:
        ts = ts.tz_localize(NY_TZ)
    else:
        ts = ts.tz_convert(NY_TZ)
    return ts

def coerce_end_date(end: Optional[pd.Timestamp]) -> pd.Timestamp:
    """If end is None, use today's date in NY. Make end inclusive at 23:59:59."""
    if end is None:
        end = today_ny()
    # include the full day for yfinance
    return end + pd.Timedelta(hours=23, minutes=59, seconds=59)

def as_business_daily(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Cast an index to business-daily frequency (B) anchored in NY time."""
    if idx.tz is None:
        idx = idx.tz_localize(NY_TZ)
    else:
        idx = idx.tz_convert(NY_TZ)
    start, end = idx.min().normalize(), idx.max().normalize()
    return pd.date_range(start, end, freq=BDay(), tz=NY_TZ)

def inner_align(*series: pd.Series) -> list[pd.Series]:
    """Align multiple series on the inner join of their indices (intersection)."""
    common = None
    for s in series:
        si = s.dropna()
        common = si.index if common is None else common.intersection(si.index)
    return [s.reindex(common).sort_index() for s in series]

def to_decimal_vol(vix_level: pd.Series) -> pd.Series:
    """Convert VIX level (in percent) to decimal implied volatility (e.g., 20 -> 0.20)."""
    return (vix_level / 100.0).rename("IV")

def ensure_tz_ny(s: pd.Series, tz: str = NY_TZ) -> pd.Series:
    if s.index.tz is None:
        s.index = s.index.tz_localize(tz)
    else:
        s.index = s.index.tz_convert(tz)
    return s
