"""Test whether higher-order GARCH models improve fit."""
import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
from itertools import product

# Load data
df = pd.read_pickle('data/processed/timeseries.pkl')
returns = df['RET_SPY'].dropna()
train_returns = returns.loc[:'2015-12-31']
returns_scaled = train_returns * 100  # Scale for numerical stability

print("Testing GARCH models with higher p, q orders...")
print("=" * 80)
print()

# Extended grid
vol_models = ['GARCH', 'EGARCH']
distributions = ['t', 'skewt']  # Skip normal (we know it's worse)
p_values = [1, 2, 3, 4, 5]
q_values = [1, 2, 3, 4, 5]

results = []
count = 0
total = len(vol_models) * len(distributions) * len(p_values) * len(q_values)

for vol, dist, p, q in product(vol_models, distributions, p_values, q_values):
    count += 1
    try:
        # Build model
        if vol == 'EGARCH':
            model = arch_model(returns_scaled, vol='EGARCH', p=p, o=1, q=q, dist=dist)
        else:
            model = arch_model(returns_scaled, vol='Garch', p=p, o=0, q=q, dist=dist)
        
        # Fit
        res = model.fit(disp='off', show_warning=False)
        
        # Check residuals
        std_resid = res.std_resid.dropna()
        arch_test = het_arch(std_resid, nlags=10)
        arch_pval = arch_test[1]
        arch_pass = arch_pval > 0.05
        
        results.append({
            'model': vol,
            'p': p,
            'q': q,
            'dist': dist,
            'BIC': res.bic,
            'AIC': res.aic,
            'ARCH_pval': arch_pval,
            'ARCH_pass': arch_pass,
            'params': len(res.params)
        })
        
        status = '✓' if arch_pass else '✗'
        print(f"[{count}/{total}] {vol}({p},{q})+{dist}: BIC={res.bic:.1f}, ARCH_p={arch_pval:.3f} {status}")
        
    except Exception as e:
        print(f"[{count}/{total}] {vol}({p},{q})+{dist}: FAILED ({str(e)[:50]})")

print()
print("=" * 80)
print("RESULTS: Top 10 models (passing ARCH test, sorted by BIC)")
print("=" * 80)

df_results = pd.DataFrame(results)
df_passed = df_results[df_results['ARCH_pass']].sort_values('BIC')

print(df_passed.head(10).to_string(index=False))

print()
print("=" * 80)
print("COMPARISON: Low vs High Order Models")
print("=" * 80)

# Compare low-order (p,q ≤ 2) vs high-order (p,q > 2)
df_passed['is_low_order'] = (df_passed['p'] <= 2) & (df_passed['q'] <= 2)

print()
print("Best LOW-order model (p,q ≤ 2):")
best_low = df_passed[df_passed['is_low_order']].iloc[0]
print(f"  {best_low['model']}({best_low['p']},{best_low['q']})+{best_low['dist']}")
print(f"  BIC: {best_low['BIC']:.2f}")
print(f"  # Parameters: {best_low['params']}")

if len(df_passed[~df_passed['is_low_order']]) > 0:
    print()
    print("Best HIGH-order model (p > 2 or q > 2):")
    best_high = df_passed[~df_passed['is_low_order']].iloc[0]
    print(f"  {best_high['model']}({best_high['p']},{best_high['q']})+{best_high['dist']}")
    print(f"  BIC: {best_high['BIC']:.2f}")
    print(f"  # Parameters: {best_high['params']}")
    
    print()
    print(f"BIC Improvement: {best_low['BIC'] - best_high['BIC']:.2f}")
    if best_high['BIC'] < best_low['BIC']:
        print("✓ Higher orders DO improve fit!")
    else:
        print("✗ Higher orders do NOT improve fit (BIC penalty dominates)")
else:
    print()
    print("No high-order models passed the ARCH test!")

