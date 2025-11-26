import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, returns_df: pd.DataFrame, costs_bps: float):
        self.returns_df = returns_df # Columns: SPY, VIX
        self.costs = costs_bps / 10000.0
        
    def run(self, target_weights: pd.DataFrame, rebalance_freq: str) -> pd.DataFrame:
        """
        Vectorized backtest with periodic rebalancing.
        """
        # Resample target weights to rebalance frequency
        # We take the weight at the end of the period to trade for the next period
        # But signals are generated at t using info up to t.
        # Trades are executed at t+1 (or close of t).
        # Let's assume we trade at the close of the signal date.
        
        # Align weights to returns index (daily)
        # Forward fill weights between rebalance dates
        
        # 1. Identify rebalance dates
        # We use 'asfreq' to get the dates, but we need to make sure they exist in the index.
        # If a rebalance date falls on a holiday/weekend (not in returns_df), we should take the previous valid date.
        
        # Generate ideal rebalance dates
        ideal_rebalance_dates = target_weights.resample(rebalance_freq).last().index
        
        # Find the closest valid trading dates (on or before)
        # We can use searchsorted or asof
        valid_dates = self.returns_df.index
        
        # Use asof to find the latest valid date for each ideal date
        # This handles holidays correctly (e.g. if Friday is holiday, trade on Thursday)
        actual_rebalance_dates = []
        for date in ideal_rebalance_dates:
            if date > valid_dates[-1]:
                continue
            loc = valid_dates.asof(date)
            if pd.notna(loc):
                actual_rebalance_dates.append(loc)
                
        actual_rebalance_dates = pd.DatetimeIndex(sorted(list(set(actual_rebalance_dates))))
        
        # 2. Create daily weights series, forward filled from rebalance dates
        # We only change weights on rebalance dates.
        # We take the target weights from the ideal dates (or the actual dates if we re-indexed target_weights)
        # target_weights is daily.
        
        # Get weights at the actual rebalance dates
        rebalance_weights = target_weights.loc[actual_rebalance_dates]
        
        # Reindex to daily and ffill
        daily_weights = rebalance_weights.reindex(self.returns_df.index).ffill()
        daily_weights = daily_weights.fillna(0.0)
        
        # 3. Calculate Portfolio Returns
        # R_p = w_spy * R_spy + w_vix * R_vix
        # Shift weights by 1 day to avoid lookahead bias (weights determined at t, apply to returns at t+1)
        
        pos_weights = daily_weights.shift(1).fillna(0.0)
        
        gross_returns = (pos_weights * self.returns_df).sum(axis=1)
        
        # 4. Calculate Turnover and Costs
        # Turnover = sum(|w_t - w_{t-1} * (1+r_{t-1})|) 
        # Simplified: sum(|w_t - w_{t-1}|)
        # We pay costs on the change in position value.
        
        weight_diff = pos_weights.diff().abs().sum(axis=1)
        costs = weight_diff * self.costs
        
        net_returns = gross_returns - costs
        
        # 5. Create Ledger
        ledger = pos_weights.copy()
        ledger["Gross_Ret"] = gross_returns
        ledger["Cost"] = costs
        ledger["Net_Ret"] = net_returns
        ledger["Equity"] = (1 + net_returns).cumprod()
        
        return ledger
