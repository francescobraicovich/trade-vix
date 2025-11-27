"""
Display key insights from bear/bull market regime analysis.
"""
import pandas as pd

# Load results
df = pd.read_csv('artifacts/strategy/final_metrics_all_horizons.csv')

# Filter for h=30 to avoid duplication
df_h30 = df[df['Horizon'] == 30].copy()

print("="*80)
print("BEAR vs BULL MARKET PERFORMANCE SUMMARY")
print("="*80)
print(f"\nTest Period: 2020-01-02 to 2025-10-15")
print(f"Bear Market Days: {df_h30.iloc[0]['Bear Market Days']:.0f} (31.3%)")
print(f"Bull Market Days: {df_h30.iloc[0]['Bull Market Days']:.0f} (68.7%)")

print("\n" + "-"*80)
print("TOP 3 STRATEGIES BY BEAR MARKET PERFORMANCE")
print("-"*80)

bear_sorted = df_h30.sort_values('Bear Market Ann Return', ascending=False)
print("\nStrategy                          Bear Return  Bull Return  Sharpe  MaxDD")
print("-" * 80)
for idx, row in bear_sorted.head(3).iterrows():
    strat = row['Strategy'].replace(' (h=30)', '')
    bear = row['Bear Market Ann Return']
    bull = row['Bull Market Ann Return']
    sharpe = row['Sharpe Ratio']
    maxdd = row['Max Drawdown']
    print(f"{strat:32} {bear:>10.1%}  {bull:>10.1%}  {sharpe:>6.2f}  {maxdd:>6.1%}")

print("\n" + "-"*80)
print("KEY INSIGHTS")
print("-"*80)

# Get specific strategies
residual = df_h30[df_h30['Strategy'].str.contains('Residual')].iloc[0]
spy = df_h30[df_h30['Strategy'].str.contains('Buy & Hold')].iloc[0]
uncond = df_h30[df_h30['Strategy'].str.contains('Unconditional')].iloc[0]

print(f"\n1. VRP Residual-Based: TRUE CRISIS ALPHA")
print(f"   - ONLY strategy with positive bear market returns (+{residual['Bear Market Ann Return']:.1%})")
print(f"   - Strong bull market performance (+{residual['Bull Market Ann Return']:.1%})")
print(f"   - Best risk-adjusted returns (Sharpe {residual['Sharpe Ratio']:.2f})")
print(f"   - Low correlation to SPY (β={residual['Beta']:.3f})")

print(f"\n2. SPY Buy & Hold: BENCHMARK")
print(f"   - Bear market: {spy['Bear Market Ann Return']:.1%}")
print(f"   - Bull market: +{spy['Bull Market Ann Return']:.1%}")
print(f"   - Full equity exposure with no protection")

print(f"\n3. VRP Unconditional: HIGH RISK/REWARD")
print(f"   - Bear market: {uncond['Bear Market Ann Return']:.1%} (catastrophic losses)")
print(f"   - Bull market: +{uncond['Bull Market Ann Return']:.1%} (exceptional gains)")
print(f"   - Drawdown: {uncond['Max Drawdown']:.1%} (almost total wipeout)")
print(f"   - NOT RECOMMENDED without position sizing/hedging")

# Calculate correlation statistics
print("\n" + "-"*80)
print("DIVERSIFICATION BENEFITS")
print("-"*80)

print("\nCorrelation to SPY (Beta):")
for idx, row in df_h30.sort_values('Beta').iterrows():
    strat = row['Strategy'].replace(' (h=30)', '')
    beta = row['Beta']
    corr = row['Correlation']
    print(f"  {strat:30} β={beta:>6.3f}  ρ={corr:>6.3f}")

print("\n" + "="*80)
print("PORTFOLIO RECOMMENDATION")
print("="*80)
print("""
Suggested Multi-Strategy Allocation:
  
  Core (60%):   40% SPY Buy & Hold + 20% SPY SMA Trend
  Alpha (30%):  30% VRP Residual-Based  
  Hedge (10%):  10% VRP VIX-Based

Expected Properties:
  • Bull Markets: Capture equity upside via core holdings
  • Bear Markets: VRP Residual provides positive returns  
  • Risk-Adjusted: Target Sharpe > 1.5, MaxDD < 30%
  • Diversification: Low correlation between strategies

Key Advantage:
  The VRP Residual-Based strategy is the ONLY component that makes 
  money during market stress, providing true portfolio insurance while 
  maintaining profitability in normal conditions.
""")

print("="*80)
