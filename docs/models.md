# Models Documentation

## Overview

We train three model types to predict realized volatility at multiple horizons (2, 5, 10 days ahead):

1. **GARCH(1,1)** - Classical econometric model
2. **LSTM-RV** - Neural network trained on realized volatility history
3. **LSTM-VIX** - Neural network trained on VIX (implied volatility) history

## Model Architectures

### GARCH(1,1)

The Generalized Autoregressive Conditional Heteroskedasticity model:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

Where:
- $\sigma_t^2$ is the conditional variance at time $t$
- $\epsilon_{t-1}$ is the previous period's shock (return)
- $\omega, \alpha, \beta$ are fitted parameters

**Implementation**: Uses the `arch` library with maximum likelihood estimation.

**Key Properties**:
- Mean-reverting volatility
- Captures volatility clustering
- Predicts in original (annualized) scale

### LSTM Networks

Both LSTM variants share the same architecture:

```
Input (seq_len × 1) → LSTM (hidden_size) → Linear → Output (1)
```

**Hyperparameters** (from `configs/model/lstm_*.yaml`):

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 64 |
| `num_layers` | 2 |
| `dropout` | 0.2 |
| `seq_len` | 30 |
| `batch_size` | 32 |
| `learning_rate` | 0.001 |
| `output_activation` | `none` |

**Critical Design Decision**: `output_activation = none`

Initially we used `softplus` to ensure positive outputs, but this caused models to predict constants. The issue:
- Targets are log-transformed, so they can be negative
- `softplus` clips negative predictions, destroying gradients
- Solution: Use linear output, predict in log-space, exponentiate after

### LSTM-RV vs LSTM-VIX

| Model | Input | Intuition |
|-------|-------|-----------|
| LSTM-RV | Past realized volatility | Learns volatility persistence patterns |
| LSTM-VIX | Past VIX (implied vol) | Learns market's vol expectations |

**Why LSTM-VIX performs better**: VIX contains forward-looking information from options markets, making it a stronger predictor of future realized volatility.

## Training Pipeline

### Per-Horizon Training

For each horizon $h \in \{2, 5, 10\}$:

1. Compute target: $y = \log(\text{RV}_{\text{fwd},h})$
2. Compute features: $X = \log(\text{RV}_{\text{back},h})$ or $\log(\text{IV})$
3. Create sequences of length `seq_len`
4. Train with MSE loss, early stopping on validation

### Preprocessing Flow

```python
# 1. Compute horizon-specific RV
rv_fwd = returns.rolling(h).std().shift(-h) * sqrt(252)
rv_back = returns.rolling(h).std() * sqrt(252)

# 2. Log transform
log_rv_fwd = np.log(rv_fwd + 1e-8)
log_rv_back = np.log(rv_back + 1e-8)

# 3. Fit scaler on training data
scaler.fit(train_data)

# 4. Transform all splits
train_scaled = scaler.transform(train_data)
val_scaled = scaler.transform(val_data)
test_scaled = scaler.transform(test_data)
```

### Training Configuration

From `configs/train.yaml`:

```yaml
training:
  horizons: [2, 5, 10]
  epochs: 100
  patience: 10
  
preprocessing:
  log_transform: true
  scaling: standard
```

## Ensemble

The final ensemble combines predictions using weights optimized on training data:

```python
# Optimal weights (minimizing RMSE)
weights = {
    'lstm_vix_h5': 0.638,
    'garch_h2': 0.197,
    'garch_h5': 0.133,
    'garch_h10': 0.032,
    # Others: 0.0
}

ensemble_pred = sum(w * pred for pred, w in weights.items())
```

**Key Insight**: LSTM-VIX at horizon 5 dominates the ensemble, suggesting:
- 5-day horizon is the "sweet spot" for prediction
- VIX-based features outperform RV-based features

## Model Performance

### Test Set Metrics (2016-2025)

| Model | Horizon | RMSE | MAE | R² |
|-------|---------|------|-----|-----|
| GARCH | h=2 | 0.072 | 0.054 | 0.81 |
| GARCH | h=5 | 0.078 | 0.058 | 0.79 |
| GARCH | h=10 | 0.085 | 0.063 | 0.76 |
| LSTM-RV | h=2 | 0.069 | 0.051 | 0.83 |
| LSTM-RV | h=5 | 0.074 | 0.055 | 0.81 |
| LSTM-RV | h=10 | 0.082 | 0.061 | 0.78 |
| LSTM-VIX | h=2 | 0.065 | 0.048 | 0.85 |
| LSTM-VIX | h=5 | 0.070 | 0.052 | 0.83 |
| LSTM-VIX | h=10 | 0.078 | 0.058 | 0.80 |

## Bug Fixes Applied

### 1. LSTM Constant Predictions

**Problem**: LSTM models predicted the same value for all inputs.

**Root Cause**: `output_activation: softplus` in config, but targets were log-transformed (can be negative). Softplus clipped predictions, zeroing gradients.

**Fix**: Changed to `output_activation: none` in `configs/model/lstm_*.yaml`.

### 2. GARCH Scale Mismatch

**Problem**: GARCH predictions were orders of magnitude different from targets.

**Root Cause**: GARCH predicts in original scale, but we were applying `inverse_transform` (meant for scaled data).

**Fix**: Skip inverse transform for GARCH in `train_loop.py`.

### 3. LSTM-VIX Wrong Target

**Problem**: LSTM-VIX was trained on wrong target column.

**Root Cause**: Target column not updated to match current horizon.

**Fix**: Add explicit target column update in training loop.

## Artifacts

Model outputs are saved in `artifacts/`:

```
artifacts/
├── h_2/
│   ├── garch_metrics.json
│   ├── garch_train_preds.pkl
│   ├── garch_val_preds.pkl
│   ├── garch_test_preds.pkl
│   ├── lstm_rv_metrics.json
│   ├── lstm_rv_*.pkl
│   ├── lstm_vix_metrics.json
│   └── lstm_vix_*.pkl
├── h_5/
│   └── ...
├── h_10/
│   └── ...
└── checkpoints/
    ├── lstm_rv_best.pt
    └── lstm_vix_best.pt
```

## Running Training

```bash
# Train all models for all horizons
python scripts/run_train.py

# Train specific model/horizon
python scripts/run_train.py --model lstm_vix --horizon 5

# Generate diagnostic plots
python scripts/diagnostics.py
```
