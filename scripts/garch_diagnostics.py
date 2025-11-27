"""
GARCH Model Diagnostics and Selection

This script performs comprehensive pre-fit diagnostics, grid search for optimal 
parameters, and post-fit validation for GARCH-family models.

Required Tests (before fitting):
1. ADF Test - Stationarity (must reject null for GARCH to be valid)
2. KPSS Test - Stationarity (must NOT reject null)
3. ARCH-LM Test - Heteroskedasticity (must reject null - ARCH effects present)
4. Ljung-Box on squared returns - Volatility clustering (must reject null)
5. Jarque-Bera - Normality (if rejected, use Student-t distribution)

Required Tests (after fitting):
1. ARCH-LM on standardized residuals - Must NOT reject (no remaining ARCH effects)
2. Ljung-Box on squared standardized residuals - Must NOT reject

Usage:
    python scripts/garch_diagnostics.py [--train-end 2015-12-31] [--val-end 2019-12-31]
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
# import seaborn as sns

warnings.filterwarnings('ignore')

# Statistical tests
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from scipy import stats

# GARCH models
from arch import arch_model
from itertools import product

# Set style for plots
plt.style.use('seaborn-v0_8-paper')
# sns.set_palette("deep")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.family'] = 'serif'

def plot_news_impact_curve(garch_res, egarch_res, output_path):
    """
    Generates and saves the News Impact Curve (NIC) comparison.
    The NIC shows how volatility at t+1 responds to a shock (return) at t.
    """
    print(f"Generating News Impact Curve comparison...")
    
    # Range of shocks (z = epsilon / sigma)
    z_range = np.linspace(-3, 3, 100)
    
    # 1. Calculate GARCH NIC
    # sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
    # We fix sigma_{t-1}^2 at the unconditional variance
    garch_params = garch_res.params
    garch_unc_var = garch_res.conditional_volatility.var() # Approximation
    
    # Extract coeffs (handling different p, q orders - taking lag 1 impact)
    omega_g = garch_params['omega']
    alpha_g = garch_params.get('alpha[1]', 0)
    beta_g = garch_params.get('beta[1]', 0)
    
    # NIC: response of sigma_t^2 to epsilon_{t-1} (where epsilon = z * sigma)
    # We plot relative to average volatility
    avg_sigma = np.sqrt(garch_unc_var)
    epsilon_range = z_range * avg_sigma
    
    # GARCH is symmetric: depends on epsilon^2
    garch_nic = omega_g + alpha_g * (epsilon_range**2) + beta_g * garch_unc_var
    
    # 2. Calculate EGARCH NIC
    # ln(sigma_t^2) = omega + alpha(|z| - E|z|) + gamma*z + beta*ln(sigma_{t-1}^2)
    egarch_params = egarch_res.params
    
    omega_e = egarch_params['omega']
    alpha_e = egarch_params.get('alpha[1]', 0)
    gamma_e = egarch_params.get('gamma[1]', 0)
    beta_e = egarch_params.get('beta[1]', 0)
    
    # E[|z|] for standard normal is sqrt(2/pi) approx 0.7979
    # For t-dist it's slightly different, but we use approx for visualization
    E_abs_z = np.sqrt(2/np.pi)
    
    # Calculate log variance response
    # We assume past log variance is at unconditional mean
    # Unconditional log variance approx: omega / (1 - beta)
    unc_log_var = omega_e / (1 - beta_e)
    
    log_var_response = omega_e + alpha_e * (np.abs(z_range) - E_abs_z) + gamma_e * z_range + beta_e * unc_log_var
    egarch_nic = np.exp(log_var_response)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(z_range, garch_nic, label='GARCH (Symmetric)', color='#1f77b4', linewidth=2, linestyle='--')
    ax.plot(z_range, egarch_nic, label='EGARCH (Asymmetric)', color='#d62728', linewidth=2)
    
    ax.set_title('News Impact Curve: GARCH vs. EGARCH', fontsize=12, fontweight='bold')
    ax.set_xlabel('Standardized Shock ($z_{t-1}$)', fontsize=10)
    ax.set_ylabel('Conditional Variance ($\sigma_t^2$)', fontsize=10)
    
    # Add annotation for leverage effect
    ax.annotate('Leverage Effect:\nNegative shocks increase\nvolatility more', 
                xy=(-2, egarch_nic[16]), xytext=(-2.5, egarch_nic[16]*1.2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    ax.legend(loc='upper center')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")


def run_pre_fit_diagnostics(returns: pd.Series, verbose: bool = True) -> dict:
    """Run all pre-fit diagnostic tests."""
    results = {}
    
    if verbose:
        print("="*70)
        print("PRE-FIT DIAGNOSTICS")
        print("="*70)
        print()
    
    # 1. ADF Test
    adf = adfuller(returns, autolag='AIC')
    results['adf'] = {
        'statistic': adf[0],
        'pvalue': adf[1],
        'lags': adf[2],
        'critical_values': adf[4],
        'pass': adf[1] < 0.05
    }
    if verbose:
        print(f"1. ADF Test: stat={adf[0]:.4f}, p={adf[1]:.6f}")
        print(f"   {'✓ PASS' if results['adf']['pass'] else '✗ FAIL'}: Returns are {'stationary' if results['adf']['pass'] else 'non-stationary'}")
        print()
    
    # 2. KPSS Test
    kpss_result = kpss(returns, regression='c', nlags='auto')
    results['kpss'] = {
        'statistic': kpss_result[0],
        'pvalue': kpss_result[1],
        'critical_values': kpss_result[3],
        'pass': kpss_result[1] > 0.05
    }
    if verbose:
        print(f"2. KPSS Test: stat={kpss_result[0]:.4f}, p={kpss_result[1]:.4f}")
        print(f"   {'✓ PASS' if results['kpss']['pass'] else '✗ FAIL'}: Returns are {'stationary' if results['kpss']['pass'] else 'non-stationary'}")
        print()
    
    # 3. ARCH-LM Test
    arch_test = het_arch(returns, nlags=10)
    results['arch_lm'] = {
        'lm_statistic': arch_test[0],
        'lm_pvalue': arch_test[1],
        'f_statistic': arch_test[2],
        'f_pvalue': arch_test[3],
        'pass': arch_test[1] < 0.05  # Want to reject H0 (no ARCH effects)
    }
    if verbose:
        print(f"3. ARCH-LM Test: LM={arch_test[0]:.4f}, p={arch_test[1]:.6f}")
        print(f"   {'✓ PASS' if results['arch_lm']['pass'] else '✗ FAIL'}: ARCH effects are {'present' if results['arch_lm']['pass'] else 'absent'}")
        print()
    
    # 4. Ljung-Box on squared returns
    lb = acorr_ljungbox(returns**2, lags=[10, 20], return_df=True)
    results['ljung_box_squared'] = {
        'statistics': lb['lb_stat'].to_dict(),
        'pvalues': lb['lb_pvalue'].to_dict(),
        'pass': (lb['lb_pvalue'] < 0.05).all()
    }
    if verbose:
        print(f"4. Ljung-Box (squared returns):")
        print(f"   Lag 10: stat={lb['lb_stat'][10]:.2f}, p={lb['lb_pvalue'][10]:.6f}")
        print(f"   Lag 20: stat={lb['lb_stat'][20]:.2f}, p={lb['lb_pvalue'][20]:.6f}")
        print(f"   {'✓ PASS' if results['ljung_box_squared']['pass'] else '✗ FAIL'}: Volatility clustering is {'present' if results['ljung_box_squared']['pass'] else 'absent'}")
        print()
    
    # 5. Jarque-Bera
    jb_stat, jb_pval = stats.jarque_bera(returns)
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    results['jarque_bera'] = {
        'statistic': jb_stat,
        'pvalue': jb_pval,
        'skewness': skewness,
        'excess_kurtosis': kurtosis,
        'non_normal': jb_pval < 0.05
    }
    if verbose:
        print(f"5. Jarque-Bera Test: stat={jb_stat:.2f}, p={jb_pval:.2e}")
        print(f"   Skewness: {skewness:.4f}, Excess Kurtosis: {kurtosis:.4f}")
        print(f"   {'Returns are NON-NORMAL (use Student-t)' if results['jarque_bera']['non_normal'] else 'Returns are approximately normal'}")
        print()
    
    # Summary
    all_pass = all([
        results['adf']['pass'],
        results['kpss']['pass'],
        results['arch_lm']['pass'],
        results['ljung_box_squared']['pass']
    ])
    results['all_pass'] = all_pass
    results['recommendation'] = 't' if results['jarque_bera']['non_normal'] else 'normal'
    
    if verbose:
        print("="*70)
        print("SUMMARY")
        print("="*70)
        print(f"All tests pass: {'✓ YES - GARCH is appropriate' if all_pass else '✗ NO'}")
        print(f"Recommended distribution: {results['recommendation']}")
    
    return results


def run_grid_search(train_returns: pd.Series, val_returns: pd.Series = None, 
                    verbose: bool = True) -> pd.DataFrame:
    """Run grid search over GARCH model specifications."""
    
    # Scale returns for numerical stability
    returns_scaled = train_returns * 100
    
    # Grid parameters
    vol_models = [
        ('GARCH', {'vol': 'Garch', 'o': 0}),
        ('EGARCH', {'vol': 'EGARCH', 'o': 1}),
    ]
    distributions = ['normal', 't', 'skewt']
    p_values = [1, 2]
    q_values = [1, 2]
    
    results = []
    
    if verbose:
        print("="*70)
        print("GRID SEARCH")
        print("="*70)
        print()
    
    for (vol_name, vol_params), dist, p, q in product(vol_models, distributions, p_values, q_values):
        try:
            # Build model
            model = arch_model(
                returns_scaled, 
                p=p, 
                q=q, 
                dist=dist,
                **vol_params
            )
            
            # Fit
            res = model.fit(disp='off', show_warning=False)
            
            # Diagnostics
            std_resid = res.std_resid.dropna()
            arch_test = het_arch(std_resid, nlags=10)
            arch_pval = arch_test[1]
            
            results.append({
                'model': vol_name,
                'p': p,
                'q': q,
                'dist': dist,
                'AIC': res.aic,
                'BIC': res.bic,
                'LogLik': res.loglikelihood,
                'ARCH_pval': arch_pval,
                'ARCH_pass': arch_pval > 0.05,
            })
            
            if verbose:
                status = '✓' if arch_pval > 0.05 else '⚠'
                print(f"  {vol_name}({p},{q})+{dist}: BIC={res.bic:.1f}, ARCH_p={arch_pval:.3f} {status}")
                
        except Exception as e:
            if verbose:
                print(f"  {vol_name}({p},{q})+{dist}: FAILED")
    
    df = pd.DataFrame(results)
    
    # Rank: prioritize models that pass ARCH test, then by BIC
    df['rank_score'] = df['BIC'] + (1 - df['ARCH_pass'].astype(int)) * 1000
    df = df.sort_values('rank_score')
    
    if verbose:
        print()
        print("="*70)
        print("TOP 5 MODELS (pass ARCH test, lowest BIC)")
        print("="*70)
        print()
        print(df[df['ARCH_pass']].head().to_string(index=False))
    
    return df


def fit_and_validate_model(returns: pd.Series, model_name: str, p: int, q: int,
                           dist: str, verbose: bool = True) -> dict:
    """Fit final model and run post-fit diagnostics."""
    
    returns_scaled = returns * 100
    
    # Determine vol type and o parameter
    if model_name == 'EGARCH':
        vol_type = 'EGARCH'
        o = 1
    elif model_name == 'GJR-GARCH':
        vol_type = 'Garch'
        o = 1
    else:  # GARCH
        vol_type = 'Garch'
        o = 0
    
    if verbose:
        print("="*70)
        print(f"FITTING {model_name}({p},{q}) with dist={dist}")
        print("="*70)
        print()
    
    # Fit model
    model = arch_model(returns_scaled, vol=vol_type, p=p, o=o, q=q, dist=dist)
    res = model.fit(disp='off')
    
    if verbose:
        print(res.summary())
        print()
    
    # Post-fit diagnostics
    std_resid = res.std_resid.dropna()
    
    # ARCH-LM
    arch_test = het_arch(std_resid, nlags=10)
    arch_pass = arch_test[1] > 0.05
    
    # Ljung-Box
    lb = acorr_ljungbox(std_resid**2, lags=[10, 20], return_df=True)
    lb_pass = (lb['lb_pvalue'] > 0.05).all()
    
    if verbose:
        print("POST-FIT DIAGNOSTICS:")
        print(f"  ARCH-LM (lag=10): p={arch_test[1]:.4f} {'✓' if arch_pass else '⚠'}")
        print(f"  Ljung-Box (lag=10): p={lb['lb_pvalue'][10]:.4f} {'✓' if lb['lb_pvalue'][10] > 0.05 else '⚠'}")
        print(f"  Ljung-Box (lag=20): p={lb['lb_pvalue'][20]:.4f} {'✓' if lb['lb_pvalue'][20] > 0.05 else '⚠'}")
        print()
    
    # Extract parameters
    params = res.params.to_dict()
    
    # Calculate persistence
    alpha_sum = sum(v for k, v in params.items() if k.startswith('alpha'))
    gamma_sum = sum(v for k, v in params.items() if k.startswith('gamma')) / 2
    beta_sum = sum(v for k, v in params.items() if k.startswith('beta'))
    persistence = alpha_sum + gamma_sum + beta_sum
    
    if verbose:
        print("PARAMETERS:")
        for k, v in params.items():
            print(f"  {k}: {v:.6f}")
        print(f"  Persistence: {persistence:.4f}")
        if persistence < 1:
            half_life = np.log(0.5) / np.log(persistence)
            print(f"  Half-life: {half_life:.1f} days")
    
    return {
        'params': params,
        'aic': res.aic,
        'bic': res.bic,
        'loglik': res.loglikelihood,
        'persistence': persistence,
        'arch_pval': arch_test[1],
        'arch_pass': arch_pass,
        'lb_pass': lb_pass,
        'result': res,
    }


def main():
    parser = argparse.ArgumentParser(description='GARCH Diagnostics and Model Selection')
    parser.add_argument('--train-end', default='2015-12-31', help='End of training period')
    parser.add_argument('--val-end', default='2019-12-31', help='End of validation period')
    parser.add_argument('--output', default='artifacts/garch_diagnostics.json', help='Output file')
    args = parser.parse_args()
    
    # Load data
    df = pd.read_pickle('data/processed/timeseries.pkl')
    returns = df['RET_SPY'].dropna()
    
    train_returns = returns.loc[:args.train_end]
    val_returns = returns.loc[args.train_end:args.val_end]
    
    print(f"Training: {train_returns.index.min().date()} to {train_returns.index.max().date()} ({len(train_returns)} obs)")
    print(f"Validation: {val_returns.index.min().date()} to {val_returns.index.max().date()} ({len(val_returns)} obs)")
    print()
    
    # 1. Pre-fit diagnostics
    pre_fit = run_pre_fit_diagnostics(train_returns)
    print()
    
    # 2. Grid search
    grid_results = run_grid_search(train_returns, val_returns)
    print()
    
    # 3. Fit best model
    # Find best of each type for comparison
    df_passed = grid_results[grid_results['ARCH_pass']]
    if df_passed.empty:
        print("WARNING: No models passed ARCH test. Using best BIC model.")
        df_passed = grid_results
        
    best_garch = df_passed[df_passed['model'] == 'GARCH'].sort_values('BIC').iloc[0]
    best_egarch = df_passed[df_passed['model'] == 'EGARCH'].sort_values('BIC').iloc[0]
    
    print(f"Best GARCH: {best_garch['model']}({best_garch['p']},{best_garch['q']})+{best_garch['dist']} (BIC={best_garch['BIC']:.1f})")
    print(f"Best EGARCH: {best_egarch['model']}({best_egarch['p']},{best_egarch['q']})+{best_egarch['dist']} (BIC={best_egarch['BIC']:.1f})")
    print()
    
    # Fit both for plotting
    garch_fit = fit_and_validate_model(
        train_returns, 'GARCH', int(best_garch['p']), int(best_garch['q']), best_garch['dist'], verbose=False
    )
    egarch_fit = fit_and_validate_model(
        train_returns, 'EGARCH', int(best_egarch['p']), int(best_egarch['q']), best_egarch['dist'], verbose=False
    )
    
    # Plot comparison
    plot_news_impact_curve(garch_fit['result'], egarch_fit['result'], Path('artifacts/plots/data_analysis/10_garch_vs_egarch_nic.png'))
    
    # Use best overall for final output
    best = df_passed.sort_values('BIC').iloc[0]
    
    final = fit_and_validate_model(
        train_returns,
        model_name=best['model'],
        p=int(best['p']), 
        q=int(best['q']), 
        dist=best['dist']
    )
    
    # Determine o and vol for config
    if best['model'] == 'EGARCH':
        vol_type = 'EGARCH'
        o = 1
    elif best['model'] == 'GJR-GARCH':
        vol_type = 'GARCH'
        o = 1
    else:
        vol_type = 'GARCH'
        o = 0
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'train_period': {'start': str(train_returns.index.min().date()), 'end': args.train_end},
        'pre_fit_diagnostics': {
            'adf_pvalue': float(pre_fit['adf']['pvalue']),
            'kpss_pvalue': float(pre_fit['kpss']['pvalue']),
            'arch_lm_pvalue': float(pre_fit['arch_lm']['lm_pvalue']),
            'jarque_bera_pvalue': float(pre_fit['jarque_bera']['pvalue']),
            'all_pass': bool(pre_fit['all_pass']),
        },
        'best_model': {
            'specification': best['model'],
            'p': int(best['p']),
            'q': int(best['q']),
            'vol': vol_type,
            'distribution': best['dist'],
            'BIC': float(best['BIC']),
        },
        'post_fit_diagnostics': {
            'arch_lm_pvalue': float(final['arch_pval']),
            'arch_pass': bool(final['arch_pass']),
            'ljung_box_pass': bool(final['lb_pass']),
        },
        'parameters': {k: float(v) for k, v in final['params'].items()},
        'persistence': float(final['persistence']),
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print()
    print(f"Results saved to: {output_path}")
    
    # Print recommended config
    print()
    print("="*70)
    print("RECOMMENDED CONFIG (configs/model/garch.yaml)")
    print("="*70)
    print()
    
    config = {
        'model_name': 'garch',
        'spec': {
            'p': int(best['p']),
            'q': int(best['q']),
            'dist': best['dist'],
            'mean': 'constant',
            'vol': vol_type,
            'horizon_days': 30,
            'annualization': 252,
        },
        'fit': {
            'update_scheme': 'static',
        }
    }
    print(yaml.dump(config, default_flow_style=False))


if __name__ == '__main__':
    main()
