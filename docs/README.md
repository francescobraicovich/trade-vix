# Trade-VIX Documentation

This documentation covers the volatility modeling and trading strategy project, which aims to predict realized volatility and develop profitable trading strategies using volatility signals.

## Documentation Structure

- **[Data](data.md)** - Data sources, preprocessing, and feature engineering
- **[Models](models.md)** - Volatility prediction models (GARCH, LSTM variants)
- **[Strategies](strategies.md)** - Trading strategy development and evolution

## Project Overview

The project explores whether we can profitably trade using volatility predictions. We train multiple models to predict realized volatility at different horizons (2, 5, 10 days ahead), then use these predictions in trading strategies.

### Key Findings

1. **Volatility Prediction**: Our ensemble model (LSTM-VIX + GARCH) achieves strong out-of-sample prediction accuracy
2. **Signal Challenge**: The VIX is structurally overpriced ~87% of the time, making the "predicted RV vs IV" signal nearly always negative
3. **Solution**: Combine a simple trend filter (SMA crossing) with volatility-based position sizing
4. **Results**: Sharpe ratio of 2.84, with 27.6% annualized alpha and max drawdown of only -7.3%

### Quick Start

```bash
# Train all models
python scripts/run_train.py

# Run backtest
python scripts/run_backtest.py

# Generate diagnostics
python scripts/diagnostics.py
```

See individual documentation files for detailed information.
