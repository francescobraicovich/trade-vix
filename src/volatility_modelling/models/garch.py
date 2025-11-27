import numpy as np
import pandas as pd
from arch import arch_model
from .base import BaseModel
from .registry import register

@register("garch")
class GarchModel(BaseModel):
    def fit(self, train_returns: pd.Series, **kwargs):
        base_spec = self.cfg["spec"]
        
        # Define grid for search
        # We search around sensible defaults for financial time series
        param_grid = {
            'p': [1, 2],
            'q': [1, 2],
            'vol': ['GARCH', 'EGARCH'],
            'dist': ['t', 'skewt'], # Financial returns are rarely normal
            'mean': ['Constant', 'Zero']
        }
        
        best_bic = float('inf')
        best_model = None
        best_res = None
        best_params = None
        
        print(f"\n[GARCH] Starting Grid Search...")
        print(f"  Grid: {param_grid}")
        
        import itertools
        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        for i, params in enumerate(combinations):
            try:
                # Skip invalid combinations if any (e.g. GARCH usually needs p>=1, q>=1)
                
                model = arch_model(
                    train_returns,
                    mean=params["mean"],
                    vol=params["vol"],
                    p=params["p"],
                    q=params["q"],
                    dist=params["dist"],
                    rescale=True
                )
                
                # Fit model
                res = model.fit(disp="off", show_warning=False)
                
                if res.bic < best_bic:
                    best_bic = res.bic
                    best_model = model
                    best_res = res
                    best_params = params
                    # print(f"  New Best: {params} | BIC: {best_bic:.2f}")
                    
            except Exception as e:
                # print(f"  Failed: {params} | Error: {e}")
                continue
                
        if best_res is None:
            raise RuntimeError("GARCH Grid Search failed to find any valid model.")
            
        self.model = best_model
        self.res = best_res
        
        print(f"\n[GARCH] Grid Search Complete.")
        print(f"  ✓ Best Parameters: {best_params}")
        print(f"  ✓ Best BIC: {self.res.bic:.2f}")
        print(f"  ✓ Scale: {self.res.scale}")
        
        # Update config with best params so predict uses them
        self.cfg["spec"].update(best_params)
        
        return self

    def predict(self, returns: pd.Series, forecast_origins: pd.DatetimeIndex, **kwargs) -> pd.Series:
        spec = self.cfg["spec"]
        horizon = spec["horizon_days"]
        annualization = spec["annualization"]
        
        # Retrieve the scale factor used during training
        scale_factor = self.res.scale
        
        # Manually scale the returns using the training scale factor
        # This ensures consistency with the fitted parameters
        scaled_returns = returns * scale_factor
        
        # Create a new model with the full data to forecast out-of-sample
        # We set rescale=False because we have already manually scaled the data
        full_model = arch_model(
            scaled_returns,
            mean=spec["mean"],
            vol=spec["vol"],
            p=spec["p"],
            q=spec["q"],
            dist=spec["dist"],
            rescale=False 
        )
        
        # Forecast using the fitted parameters
        # We set start=0 to generate forecasts for the entire history
        # EGARCH requires simulation for horizon > 1
        method = 'analytic'
        if spec['vol'] == 'EGARCH' and horizon > 1:
            method = 'simulation'
            
        forecasts = full_model.forecast(
            self.res.params, 
            horizon=horizon, 
            start=0, 
            reindex=False,
            method=method,
            simulations=1000
        )
        
        # forecasts.variance is a DataFrame with columns h.1, h.2, ... h.30
        # index is the time t (origin).
        
        # We filter for the requested origins
        valid_origins = forecasts.variance.index.intersection(forecast_origins)
        variance_paths = forecasts.variance.loc[valid_origins]
        
        # Adjust for scaling
        # The variance output is for the scaled data, so it is scaled by scale_factor^2
        # We need to divide by scale_factor^2 to get back to the original scale
        variance_paths = variance_paths / (scale_factor ** 2)
        
        # Sum variances over horizon
        total_variance = variance_paths.sum(axis=1)
        
        # Convert to annualized volatility
        # rv30_hat = sqrt((252/30) * sum_{i=1..30} variance_{t+i})
        rv30_hat = np.sqrt((annualization / horizon) * total_variance)
        
        return rv30_hat
