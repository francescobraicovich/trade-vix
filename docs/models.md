# Models & Signal Generation Report

## 1. Modeling Objective: The Search for "Fair Value"

The core hypothesis of this project is that the market price of volatility (VIX) often deviates from its "fair value" (future Realized Volatility). By accurately predicting Realized Volatility (RV), we can identify these mispricings and capture the **Volatility Risk Premium (VRP)**.

$$ \text{VRP} = \text{Implied Volatility (VIX)} - \text{Predicted Realized Volatility (RV)} $$

We employ a multi-model approach to estimate this fair value, combining econometric rigor with deep learning flexibility.

---

## 2. Data & Features

For a detailed explanation of the data sources, feature engineering, and lookahead bias prevention, please refer to the **[Data Methodology Report](data.md)**.

In summary:
*   **Target**: Forward Realized Volatility ($RV_{fwd, h}$) over horizon $h$.
*   **Features**: Past Realized Volatility ($RV_{back}$) and Implied Volatility ($VIX$).
*   **Preprocessing**: All volatility variables are log-transformed to stabilize variance and ensure positive predictions.

---

## 3. The Model Zoo

We train three distinct model architectures to capture different market dynamics. Each represents a different "philosophy" of volatility modeling.

### 3.1 GARCH & EGARCH (The Econometric Approach)

**Generalized Autoregressive Conditional Heteroskedasticity (GARCH)** is the gold standard in financial econometrics. It treats volatility not as a constant, but as a time-varying process that exhibits *clustering* (large moves follow large moves).

#### 3.1.1 Standard GARCH: The Symmetric Baseline
The standard GARCH(1,1) model assumes that today's variance ($\sigma_t^2$) depends on three things:
1.  **Long-run average variance** ($\omega$).
2.  **Yesterday's shock** (squared return, $\epsilon_{t-1}^2$).
3.  **Yesterday's variance** ($\sigma_{t-1}^2$).

$$ \sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2 $$

*   **Intuition**: If the market moved wildly yesterday (large $\epsilon_{t-1}^2$), volatility increases today. If volatility was already high yesterday (large $\sigma_{t-1}^2$), it likely remains high today (persistence).
*   **The Flaw**: It squares the return ($\epsilon_{t-1}^2$). This means a +5% rally and a -5% crash have the *exact same effect* on predicted volatility.

#### 3.1.2 EGARCH: Capturing the "Fear Factor"
Financial markets are not symmetric. As shown in our **[Data Report](data.md#75-leverage-effect)**, there is a strong **Leverage Effect**: volatility spikes significantly more during market crashes than during rallies.

**Exponential GARCH (EGARCH)** fixes the symmetry flaw by modeling the *logarithm* of variance:

$$ \ln(\sigma_t^2) = \omega + \beta \ln(\sigma_{t-1}^2) + \alpha \left( \left| \frac{\epsilon_{t-1}}{\sigma_{t-1}} \right| - \sqrt{\frac{2}{\pi}} \right) + \gamma \frac{\epsilon_{t-1}}{\sigma_{t-1}} $$

*   **The $\gamma$ Term (Asymmetry)**: This is the critical innovation.
    *   If $\gamma < 0$ (which is typical for equities), negative returns ($\epsilon_{t-1} < 0$) increase volatility *more* than positive returns.
    *   This aligns with the "panic" dynamic: fear drives volatility higher than greed.
*   **Log-Form**: By modeling $\ln(\sigma^2)$, we ensure that the predicted variance is always positive, without needing artificial constraints on parameters.

**Our Implementation**: We perform a **Grid Search** to select the best specification. Our data consistently selects **EGARCH(2,1) with a Skewed Student-t distribution**, confirming that the market exhibits both leverage effects and fat tails (extreme events happen more often than a Normal distribution predicts).

---

### 3.2 Recurrent Neural Networks & LSTMs (The Deep Learning Approach)

While GARCH models are rigorous, they are rigid. They assume volatility follows a strict mathematical formula. **Recurrent Neural Networks (RNNs)** relax this assumption, learning the "laws of physics" of volatility directly from data.

#### 3.2.1 The RNN Concept
A standard Feed-Forward Neural Network sees data as independent snapshots. An RNN, however, has **memory**. It processes data sequentially ($t=1, t=2, \dots$), maintaining a "hidden state" ($h_t$) that acts as a summary of everything it has seen so far.

$$ h_t = \tanh(W_x x_t + W_h h_{t-1} + b) $$

*   **Intuition**: When predicting volatility for today, the RNN considers not just today's features ($x_t$), but also the context of the last 60 days ($h_{t-1}$).

#### 3.2.2 LSTM: Solving the "Goldfish Memory" Problem
Standard RNNs suffer from the **Vanishing Gradient Problem**: they tend to forget information from long ago (e.g., a regime change 50 days ago might be lost).

**Long Short-Term Memory (LSTM)** networks introduce a "gating" mechanism to control the flow of information:
1.  **Forget Gate**: "What irrelevant history should I throw away?"
2.  **Input Gate**: "What new information is worth storing?"
3.  **Output Gate**: "What part of my memory is relevant for the prediction *right now*?"

This allows the LSTM to maintain a "long-term memory" cell ($C_t$) that can carry critical information (like "we are in a high-volatility crisis regime") over long sequences without degradation.

#### 3.2.3 Our Architectures
We deploy two specialized LSTMs:

1.  **LSTM-RV (The Historian)**:
    *   **Input**: Past Realized Volatility.
    *   **Mechanism**: A **Bidirectional LSTM**. It reads the 60-day history both forwards (past $\to$ present) and backwards (present $\to$ past). This allows it to better understand the *structure* of the volatility trend.
    *   **Role**: Captures complex, non-linear autocorrelation patterns that GARCH misses.

2.  **LSTM-VIX (The Translator)**:
    *   **Input**: Implied Volatility (VIX).
    *   **Mechanism**: Bidirectional LSTM.
    *   **Role**: VIX is a biased estimator (it includes a risk premium). The LSTM learns the non-linear mapping function $f(VIX) \to RV$, effectively "de-biasing" the market's expectation to find the true fair value.

---

## 4. Why Three Models? The "Ensemble of Views"

We deliberately use three distinct models because they suffer from different "blind spots". By combining them, we create a more robust signal.

1.  **GARCH (The Theorist)**:
    *   *Strength*: Mathematically guaranteed to capture the leverage effect and mean reversion. Very stable in "normal" markets.
    *   *Weakness*: Rigid. It assumes volatility follows a strict formula. If the market regime changes (e.g., a new type of crisis), GARCH is slow to adapt.

2.  **LSTM-RV (The Historian)**:
    *   *Strength*: Flexible. It learns patterns purely from data without assuming a formula. It can capture complex, non-linear dependencies that GARCH misses.
    *   *Weakness*: Backward-looking. It only knows what *has* happened, not what the market *expects* to happen.

3.  **LSTM-VIX (The Sentiment Reader)**:
    *   *Strength*: Forward-looking. It sees the "fear" in the market (via option prices) before it materializes in returns.
    *   *Weakness*: Biased. VIX often overreacts to news. The LSTM learns to "de-bias" this signal, but it is still sensitive to market sentiment shocks.

**Conclusion**:
*   When **GARCH** and **LSTM-RV** agree, the historical trend is strong.
*   When **LSTM-VIX** diverges from the others, the market is pricing in a *future* event that hasn't happened yet.
*   Combining them gives us a "Fair Value" that respects history but listens to the market.

---

## 5. Training Protocol

### 4.1 Walk-Forward Validation
We split the data chronologically to strictly prevent lookahead bias:
*   **Train**: 1990 - 2015
*   **Validation**: 2016 - 2019 (Used for early stopping and hyperparameter tuning)
*   **Test**: 2020 - 2025 (Truly out-of-sample)

### 4.2 Leakage Prevention
A critical detail in volatility modeling is the overlap between input and target windows.
*   At time $t$, we predict $RV$ over $[t+1, t+h]$.
*   To ensure zero leakage, the training set must end at least $h$ days before the validation set begins. We enforce a buffer gap to ensure no target in the training set uses data from the validation period.



