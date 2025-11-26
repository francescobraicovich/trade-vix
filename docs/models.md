# Models & Signal Generation Report

## 1. Modeling Objective

The primary goal of the modeling phase is **not** just to predict volatility, but to establish a **fair value** for Realized Volatility (RV). By accurately predicting what RV *should* be, we can compare it to the market price of volatility (VIX) to identify mispricings.

$$ \text{Volatility Risk Premium (VRP)} = VIX - \text{Predicted RV} $$

We employ two distinct modeling paradigms to capture different aspects of market behavior:
1.  **Econometric (EGARCH)**: Captures stylized facts like leverage effects and mean reversion.
2.  **Deep Learning (LSTM)**: Captures complex, non-linear temporal dependencies.

---

## 2. The Three Model Types

We train three distinct types of models. Each "views" the market through a different lens:

### 2.1 EGARCH (Exponential GARCH)
*   **Input**: Past Daily Returns ($r_{t-1}, r_{t-2}, \dots$)
*   **Mechanism**: A statistical model that assumes variance follows a specific mathematical process.
*   **Why it works**: It explicitly models the **Leverage Effect** (negative returns increase volatility more than positive ones) and **Fat Tails** (extreme events happen more often than a Normal distribution predicts).
*   **Role**: Provides a robust, theoretically grounded baseline. It is less prone to overfitting than neural networks.

### 2.2 LSTM-RV (Pure Time-Series)
*   **Input**: Past Realized Volatility ($\ln(RV_{t-1}), \dots, \ln(RV_{t-60})$)
*   **Mechanism**: A Recurrent Neural Network (RNN) that learns patterns in the volatility series itself.
*   **Why it works**: Volatility is highly persistent (if it's high today, it's likely high tomorrow). The LSTM captures complex autocorrelation structures that linear models miss.
*   **Role**: Captures the "momentum" of volatility.

### 2.3 LSTM-VIX (Market-Implied)
*   **Input**: Past Implied Volatility ($\ln(VIX_{t}), \dots, \ln(VIX_{t-59})$)
*   **Mechanism**: An LSTM trained to predict *Realized* Volatility using *Implied* Volatility as input.
*   **Why it works**: VIX is determined by option prices, which incorporate traders' expectations of the future. It contains "forward-looking" information that historical returns (used by GARCH/LSTM-RV) do not have.
*   **Role**: Incorporates market sentiment and fear into the prediction.

---

## 3. The "9 Predictions" Strategy

We do not just train one model. We train all 3 model types across **3 distinct time horizons**, resulting in a matrix of 9 predictions:

| Model Type | Horizon: 2 Days | Horizon: 5 Days | Horizon: 10 Days |
| :--- | :---: | :---: | :---: |
| **EGARCH** | $P_{1}$ | $P_{2}$ | $P_{3}$ |
| **LSTM-RV** | $P_{4}$ | $P_{5}$ | $P_{6}$ |
| **LSTM-VIX** | $P_{7}$ | $P_{8}$ | $P_{9}$ |

### Why Multiple Horizons?
Volatility has a **Term Structure**. A shock (e.g., a Fed announcement) might spike volatility for 2 days but not 10. By analyzing multiple horizons, we determine the optimal "trading frequency" for our strategy.

*   **2 Days**: Too noisy. High transaction costs would eat up profits.
*   **10 Days**: Too smooth. The signal lags behind the market.
*   **5 Days (Winner)**: The "Sweet Spot". It balances signal stability with responsiveness.

---

## 4. Signal Generation Pipeline

The final trading signal is constructed in three steps, filtering the 9 raw predictions down to a single actionable metric.

### Step 1: Ensemble Construction (The "Fair Value")
We discard the noisy (2d) and laggy (10d) models. We combine the best 5-day models to create a robust "Fair Value" estimate for volatility.

$$ \ln(\widehat{RV}_{fair}) = 0.64 \times \ln(\widehat{RV}_{LSTM-VIX}) + 0.36 \times \ln(\widehat{RV}_{EGARCH}) $$

*   *Note*: We combine them in log-space (Geometric Mean).
*   *Weights*: Derived from validation set performance. The LSTM-VIX gets higher weight because VIX is a superior predictor, but EGARCH provides stability.

### Step 2: Calculate Raw VRP
We compare the market price (VIX) to our Fair Value estimate.

$$ VRP_{raw} = VIX - \widehat{RV}_{fair} $$

*   Usually, $VRP_{raw} > 0$ (Insurance is expensive).
*   If we just sold when $VRP > 0$, we would sell constantly, even during crashes.

### Step 3: Residual Filtering (The "Alpha")
This is the critical innovation. We recognize that **VRP naturally expands when VIX is high**. A high premium during a crisis is *justified* risk compensation, not a mispricing.

We model the "Justified VRP" using a dynamic regression:
$$ \text{Justified VRP}_t = \alpha + \beta \times VIX_t $$

The **Trading Signal** is the *unexplained* portion (the residual):

$$ \text{Signal}_t = VRP_{raw, t} - \text{Justified VRP}_t $$

*   **Signal > 0**: The premium is *abnormally* high, even accounting for the current panic. **SELL VOLATILITY.**
*   **Signal < 0**: The premium is normal or low. **STAY CASH.**

This ensures we only sell insurance when it is mathematically overpriced, not just when it is expensive.
