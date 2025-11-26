from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import pandas as pd

@dataclass
class DateSplits:
    train_end: str      # inclusive
    valid_end: str      # inclusive
    test_end: str       # inclusive

def split_by_date(df: pd.DataFrame, splits: DateSplits) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = df.index
    
    train_end = pd.to_datetime(splits.train_end)
    valid_end = pd.to_datetime(splits.valid_end)
    test_end = pd.to_datetime(splits.test_end)
    
    if idx.tz is not None:
        # If the index is tz-aware, we need to localize the split dates.
        # However, if the split dates are just dates (YYYY-MM-DD), tz_localize might set time to 00:00:00.
        # If the index has specific times, this is fine as long as we are consistent.
        # But if the index is America/New_York, we should probably use that.
        train_end = train_end.tz_localize(idx.tz)
        valid_end = valid_end.tz_localize(idx.tz)
        test_end = test_end.tz_localize(idx.tz)

    train = df.loc[idx <= train_end]
    valid = df.loc[(idx > train_end) & (idx <= valid_end)]
    test  = df.loc[(idx > valid_end)  & (idx <= test_end)]
    return train, valid, test
