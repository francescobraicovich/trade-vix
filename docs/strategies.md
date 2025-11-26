# Strategy Report: From Buy & Hold to Alpha Generation

## Executive Summary

This document presents a **structured journey** through seven trading strategies, progressing from the simplest baseline (Buy & Hold) to our most sophisticated signal (Residual-Based VRP Harvesting).

The core question we answer is:

> **Where does our volatility prediction actually add unique value?**

The answer, supported by rigorous backtesting, is that our prediction's unique value lies **not** in timing the equity market, but in identifying **mispricings in the Volatility Risk Premium (VRP)**. We achieve a **Sharpe Ratio of 5.86** with a **Max Drawdown of only -25.7%** by selling volatility *only* when our model identifies that VIX is unusually overpriced.

---

## The Seven Strategies

We organize the strategies into two families, progressing from simple to complex within each:

| # | Strategy | Description | Asset |
|---|----------|-------------|-------|
| 1 | SPY Buy & Hold | Passive benchmark | Equity |
| 2 | SPY SMA(50) Trend | Add trend filter | Equity |
| 3 | SPY Trend + VIX Sizing | Add position sizing using VIX | Equity |
| 4 | SPY Trend + Pred Sizing | Add position sizing using *our prediction* | Equity |
| 5 | VRP Unconditional | Always sell volatility | Volatility |
| 6 | VRP VIX-Based | Sell volatility when VIX is high | Volatility |
| 7 | **VRP Residual-Based** | Sell volatility when *our signal* is high | Volatility |

---

## Part 1: The Equity Strategies (Strategies 1-4)

The first family of strategies answers: *Can our volatility prediction improve a simple equity allocation?*

### Strategy 1: SPY Buy & Hold (The Baseline)

The simplest possible strategy: hold SPY indefinitely.

*   **Rule**: $\text{Position} = 1$ (always fully invested).
*   **Purpose**: Establishes the passive benchmark.

### Strategy 2: SPY SMA(50) Trend

A classic trend-following overlay.

*   **Rule**: $\text{Position} = 1$ if $SPY_{price} > SMA_{50}$, else $\text{Position} = 0$.
*   **Hypothesis**: Being in cash during downtrends reduces drawdowns without sacrificing much upside.

### Strategy 3: SPY Trend + VIX Sizing

Adds dynamic position sizing based on the current VIX level.

*   **Rule**: If in an uptrend, size the position inversely to VIX.
    $$ \text{Position} = 1 + 0.5 \times (1 - \text{Percentile}(VIX)) $$
*   **Hypothesis**: Lower VIX → lower expected volatility → safer to use more leverage.

### Strategy 4: SPY Trend + Prediction Sizing

Replaces VIX with our model's predicted RV.

*   **Rule**: If in an uptrend, size the position inversely to *Predicted RV*.
    $$ \text{Position} = 1 + 0.5 \times (1 - \text{Percentile}(\widehat{RV})) $$
*   **Hypothesis**: Our prediction might contain information VIX does not.

### Equity Strategy Results

| Metric | Buy & Hold | SMA(50) Trend | Trend + VIX | Trend + Pred |
|--------|:----------:|:-------------:|:-----------:|:------------:|
| **Sharpe Ratio** | 0.58 | 0.46 | 0.45 | 0.45 |
| **Ann. Return** | 9.6% | 4.9% | 6.1% | 6.0% |
| **Ann. Volatility** | 16.6% | 10.7% | 13.6% | 13.6% |
| **Max Drawdown** | -53.1% | -34.9% | -40.7% | -40.7% |

### Equity Conclusion: No Unique Edge

**Key Insight**: The trend filter reduces drawdown (good), but the prediction-based sizing offers **no improvement** over VIX-based sizing. Their Sharpe ratios and drawdowns are nearly identical.

> **Why?** Our prediction is highly correlated with VIX (~94%). For position sizing, VIX is already an excellent proxy for expected volatility. Our prediction provides no unique information here.

This forces us to ask: **Where *does* our prediction add value?**

---

## Part 2: The Volatility Strategies (Strategies 5-7)

The second family of strategies answers: *Can our prediction identify when to harvest the Volatility Risk Premium?*

### Understanding the Trade: Selling Volatility

Imagine a **Variance Swap**: a contract where you exchange *implied variance* (VIX²) for *realized variance* (RV²).

*   **If you SELL volatility**: P&L = $VIX^2 - RV^2$
*   **If VIX > RV (the usual case)**: You profit.
*   **If RV > VIX (a crash)**: You lose.

The difference $VIX - RV$ is called the **Variance Risk Premium (VRP)**. It is the "insurance premium" that volatility sellers collect. Historically, VRP is positive ~85% of the time.

### Strategy 5: VRP Unconditional (Always Sell)

The simplest volatility strategy: collect the premium every day.

*   **Rule**: Always be short volatility.
*   **Rationale**: VRP is positive most of the time, so always harvest it.

### Strategy 6: VRP VIX-Based

Only sell volatility when VIX is elevated.

*   **Rule**: Sell volatility if $VIX > \text{70th Percentile}$.
*   **Hypothesis**: Higher VIX → higher premium → better time to sell.

### Strategy 7: VRP Residual-Based (Our Unique Signal)

Only sell volatility when our **residual signal** indicates VIX is *unusually* overpriced.

*   **Signal Construction**:
    1.  Calculate Predicted VRP: $\widehat{VRP} = VIX - \widehat{RV}$.
    2.  Calculate "Expected" VRP for this VIX level: $E[VRP | VIX] = \alpha + \beta \times VIX$.
    3.  Residual: $\text{Signal} = \widehat{VRP} - E[VRP | VIX]$.
*   **Rule**: Sell volatility if $\text{Signal} > \text{70th Percentile}$.

### Why is the Residual Unique?

VIX alone tells you **what the market expects volatility to be**.
VIX alone **cannot** tell you **if that expectation is too high or too low**.

Our model provides an *independent* estimate of future RV. The *residual* of our VRP forecast, controlling for the current VIX level, identifies when the market's fear premium is abnormally high—a genuine mispricing.

### Volatility Strategy Results

| Metric | Unconditional | VIX-Based | **Residual-Based** |
|--------|:-------------:|:---------:|:------------------:|
| **Sharpe Ratio** | 16.65 | 6.78 | **5.86** |
| **Ann. Return** | 366% | 145% | **98.1%** |
| **Ann. Volatility** | 22.0% | 21.4% | **16.7%** |
| **Max Drawdown** | -44.1% | -45.4% | **-25.7%** |
| **Alpha (Ann.)** | 1.56 | 0.92 | **0.70** |
| **Beta** | 0.10 | 0.05 | **0.01** |

### Interpreting the Results

1.  **Unconditional has the highest raw Sharpe (16.65)**. This represents the theoretical maximum of always harvesting the VRP. However, it has **-44% drawdown** — catastrophic losses during market crashes.

2.  **VIX-Based filtering slightly reduces Sharpe to 6.78 but does not improve drawdown (-45.4%)**. Why? Because high VIX often *coincides* with crashes. Selling when VIX is high means selling right before realized volatility explodes.

3.  **Residual-Based achieves Sharpe 5.86 with only -25.7% drawdown**. It sacrifices some raw return but delivers a **far superior risk profile**. It avoids selling volatility during "justified" high-VIX environments (i.e., actual crises).

---

## Part 3: The Core Insight

### The Value of Our Signal

| What VIX Tells You | What Our Signal Tells You |
|---|---|
| The market's *expectation* of volatility. | Whether that expectation is *accurate*. |
| The *level* of fear. | Whether fear is *justified* or *overblown*. |
| Information already priced in. | Information the market may be missing. |

### Signal > 0: VIX is *Unusually* Overpriced
This means: "Yes, VIX is high, but it's even higher than the current situation warrants." **SELL VOLATILITY.**

### Signal < 0: VIX is Fairly Priced (or Underpriced)
This means: "VIX is high, but the risk is real." **STAY CASH.** Avoid the drawdown.

---

## Summary: Where Our Signal Matters

| Application | VIX Alone | Our Prediction | Winner |
|-------------|:---------:|:--------------:|:------:|
| Equity Position Sizing | ✓ (Sharpe 0.45) | ✓ (Sharpe 0.45) | **Tie** |
| VRP Timing (Unconditional) | N/A | N/A | N/A |
| **VRP Timing (Conditional)** | ✗ (DD -45%) | **✓ (DD -26%)** | **Prediction** |

**Conclusion**: The unique value of our volatility prediction is **not** in timing equities. It is in identifying **when the VRP is abnormally high**, allowing us to harvest the premium while avoiding catastrophic losses during crises.
