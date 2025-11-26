import numpy as np
import pandas as pd
from arch import arch_model
from .base import BaseModel
from .registry import register

@register("garch")
class GarchModel(BaseModel):
    def fit(self, train_returns: pd.Series, **kwargs):
        spec = self.cfg["spec"]
        # arch_model expects returns in percentage or similar scale often, but log returns are fine.
        # Usually better to scale up by 100 for numerical stability if they are raw log returns.
        # But prompt says "RET_SPY (daily log returns)". I will use as is or scale if needed.
        # Standard practice with arch is often returns * 100. 
        # However, the target RV30 is annualized. 
        # If I use raw log returns, variance is in raw units.
        # rv30_hat = sqrt((252/30) * sum(variance))
        # If returns are raw, variance is raw. sqrt(variance) is raw vol. * sqrt(252) is annualized.
        # So it matches.
        
        self.model = arch_model(
            train_returns,
            mean=spec["mean"],
            vol=spec["vol"],
            p=spec["p"],
            q=spec["q"],
            dist=spec["dist"],
            rescale=False # Important to keep units consistent
        )
        self.res = self.model.fit(disp="off", show_warning=False)
        return self

    def predict(self, returns: pd.Series, forecast_origins: pd.DatetimeIndex, **kwargs) -> pd.Series:
        spec = self.cfg["spec"]
        horizon = spec["horizon_days"]
        annualization = spec["annualization"]
        
        # Create a new model with the full data to forecast out-of-sample
        # We use the same specification as the fitted model
        full_model = arch_model(
            returns,
            mean=spec["mean"],
            vol=spec["vol"],
            p=spec["p"],
            q=spec["q"],
            dist=spec["dist"],
            rescale=False 
        )
        
        # Forecast using the fitted parameters
        # We set start=0 to generate forecasts for the entire history
        forecasts = full_model.forecast(self.res.params, horizon=horizon, start=0, reindex=False)
        
        # forecasts.variance is a DataFrame with columns h.1, h.2, ... h.30
        # index is the time t (origin).
        
        # We filter for the requested origins
        valid_origins = forecasts.variance.index.intersection(forecast_origins)
        variance_paths = forecasts.variance.loc[valid_origins]
        
        # Sum variances over horizon
        total_variance = variance_paths.sum(axis=1)
        
        # Convert to annualized volatility
        # rv30_hat = sqrt((252/30) * sum_{i=1..30} variance_{t+i})
        rv30_hat = np.sqrt((annualization / horizon) * total_variance)
        
        return rv30_hat
