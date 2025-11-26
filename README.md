# Trade-VIX: Volatility Risk Premium Alpha Generation

A quantitative trading system that predicts Realized Volatility (RV) using EGARCH and LSTM models, then generates trading signals by identifying mispricings in the **Volatility Risk Premium (VRP)**.

## Key Result

| Strategy | Sharpe Ratio | Max Drawdown | Ann. Alpha |
|----------|:------------:|:------------:|:----------:|
| SPY Buy & Hold | 0.58 | -53.1% | 0% |
| VRP Residual-Based | **5.86** | **-25.7%** | **70%** |

> **Core Insight**: Our prediction's unique value lies *not* in timing equities, but in identifying when VIX is **unusually overpriced** relative to predicted RV—allowing us to harvest the Variance Risk Premium while avoiding catastrophic losses during crises.

---

## How It Works

### 1. Data Pipeline
- **Source**: SPY (returns) and VIX (implied volatility) from Yahoo Finance (1993–present).
- **Target**: 30-day Forward Realized Volatility (strict lookahead-bias prevention).
- **Preprocessing**: Log-transform, winsorization (1%/99%), temporal train/val/test splits.

### 2. Models (3 Types × 3 Horizons = 9 Predictions)
| Model | Input | Why It Works |
|-------|-------|--------------|
| **EGARCH(2,1)** | Returns | Captures leverage effect & fat tails |
| **LSTM-RV** | Past RV | Captures volatility momentum |
| **LSTM-VIX** | Past VIX | Incorporates forward-looking market sentiment |

### 3. Signal Generation
```
Step 1: Ensemble → Fair Value RV (64% LSTM-VIX + 36% EGARCH at 5-day horizon)
Step 2: VRP_raw = VIX - Fair Value RV
Step 3: Signal = VRP_raw - E[VRP | VIX]  (the "residual")

If Signal > 70th percentile → SELL VOLATILITY
Else → STAY CASH
```

The **residual** filters out "justified" high premiums during actual crises, leaving only genuine mispricings.

---

## Project Structure

```
trade-vix/
├── configs/                 # YAML configurations
│   ├── data.yaml
│   ├── train.yaml
│   └── model/              # Per-model configs (garch.yaml, lstm_*.yaml)
├── data/
│   ├── raw/                # Downloaded data
│   └── processed/          # timeseries.pkl
├── src/volatility_modelling/
│   ├── data/               # Loaders, features, preprocessing
│   ├── models/             # EGARCH, LSTM implementations
│   ├── strategy/           # Backtesting & signal logic
│   └── training/           # Training loop, callbacks
├── scripts/
│   ├── download_data.py    # Fetch SPY/VIX data
│   ├── run_train.py        # Train all models
│   └── run_final_pipeline.py  # End-to-end execution
├── artifacts/              # Model checkpoints, metrics, plots
└── docs/
    ├── data.md             # Data & methodology report
    ├── models.md           # Model architecture & signal generation
    └── strategies.md       # Strategy comparison & results
```

---

## Quick Start

```bash
# 1. Setup environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Download data
python scripts/download_data.py

# 3. Run full pipeline (train + backtest)
python scripts/run_final_pipeline.py
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/data.md](docs/data.md) | Data sources, feature engineering, statistical tests (ADF, ARCH-LM, Jarque-Bera) |
| [docs/models.md](docs/models.md) | EGARCH/LSTM architecture, the "9 predictions" framework, signal construction |
| [docs/strategies.md](docs/strategies.md) | Strategy comparison (7 strategies), why the residual signal outperforms |

---

## Results Summary

### Why Residual-Based Wins

| What VIX Tells You | What Our Signal Tells You |
|--------------------|---------------------------|
| Market's *expectation* of volatility | Whether that expectation is *accurate* |
| The *level* of fear | Whether fear is *justified* or *overblown* |

**VIX alone cannot tell you if VIX itself is overpriced.** Our independent RV prediction enables this decomposition.

---

## Requirements

- Python 3.9+
- PyTorch, arch, pandas, numpy, yfinance
- See `requirements.txt` for full list

---

## License

MIT