
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import os

# Set style for professional/academic plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("deep")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.family'] = 'serif' # Academic look

OUTPUT_DIR = "artifacts/plots/data_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    print("Loading data...")
    df = pd.read_pickle("data/processed/timeseries.pkl")
    # Load raw prices for context if available, otherwise reconstruct from returns roughly or just use VIX
    # We have raw/sp500.pkl and raw/vix.pkl
    try:
        spy_px = pd.read_pickle("data/raw/sp500.pkl")
        vix_px = pd.read_pickle("data/raw/vix.pkl")
    except:
        spy_px = None
        vix_px = None
    return df, spy_px, vix_px

def plot_market_overview(spy_px, vix_px):
    print("Generating Market Overview plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # SPY Price
    if spy_px is not None:
        ax1.plot(spy_px.index, spy_px.values, color='#1f77b4', linewidth=1)
        ax1.set_title('S&P 500 Price History (SPY)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Price ($)')
        ax1.grid(True, alpha=0.3)
    
    # VIX
    if vix_px is not None:
        ax2.plot(vix_px.index, vix_px.values, color='#d62728', linewidth=1)
        ax2.set_title('CBOE VIX Index (Implied Volatility)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('VIX Level')
        ax2.grid(True, alpha=0.3)
        # Add long-term mean
        mean_vix = vix_px.mean()
        ax2.axhline(mean_vix, color='black', linestyle='--', alpha=0.5, label=f'Mean: {mean_vix:.1f}')
        ax2.legend()

    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_market_overview.png")
    plt.close()

def plot_volatility_clustering(returns):
    print("Generating Volatility Clustering plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Raw Returns
    ax1.plot(returns.index, returns.values, color='black', linewidth=0.5, alpha=0.8)
    ax1.set_title('Daily Log Returns (SPY)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Log Return')
    ax1.grid(True, alpha=0.3)
    
    # Squared Returns (Proxy for Volatility)
    ax2.plot(returns.index, returns.values**2, color='#ff7f0e', linewidth=0.5, alpha=0.8)
    ax2.set_title('Squared Returns (Volatility Clustering)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Squared Return')
    ax2.grid(True, alpha=0.3)
    
    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_volatility_clustering.png")
    plt.close()

def plot_distribution_analysis(returns):
    print("Generating Distribution Analysis plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram of returns
    sns.histplot(returns, stat='density', bins=100, color='gray', alpha=0.4, label='Empirical Data', ax=ax)
    
    # Fit Normal Distribution
    mu, std = stats.norm.fit(returns)
    x = np.linspace(returns.min(), returns.max(), 1000)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(x, p, 'r--', linewidth=2, label=f'Normal Dist ($\mu={mu:.4f}, \sigma={std:.4f}$)')
    
    # Fit T-Distribution (Closer to reality)
    params = stats.t.fit(returns)
    p_t = stats.t.pdf(x, *params)
    ax.plot(x, p_t, 'b-', linewidth=2, label='Student-t Dist (Fat Tails)')

    ax.set_title('Return Distribution vs. Normal Assumption', fontsize=12, fontweight='bold')
    ax.set_xlabel('Daily Log Return')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_xlim(-0.05, 0.05) # Zoom in on the center to show peak
    ax.grid(True, alpha=0.3)
    
    # Add text box with stats
    kurt = stats.kurtosis(returns)
    skew = stats.skew(returns)
    textstr = '\n'.join((
        f'Skewness: {skew:.2f}',
        f'Kurtosis: {kurt:.2f} (Excess)',
        r'Result: Non-Normal (Fat Tails)'
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_distribution_fat_tails.png")
    plt.close()

def plot_autocorrelation(returns):
    print("Generating ACF/PACF plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ACF of Raw Returns (Should be near zero -> Random Walk)
    plot_acf(returns, lags=40, ax=ax1, title='ACF of Raw Returns (Mean Independence)')
    ax1.set_ylim(-0.2, 0.2)
    
    # ACF of Squared Returns (Should be significant -> Volatility Dependence)
    plot_acf(returns**2, lags=40, ax=ax2, title='ACF of Squared Returns (Volatility Dependence)')
    ax2.set_ylim(-0.1, 0.5)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_autocorrelation_check.png")
    plt.close()

def plot_leverage_effect(returns):
    print("Generating Leverage Effect plot...")
    # Leverage effect: Correlation between returns and subsequent volatility
    # We use squared returns as proxy for volatility
    
    df = pd.DataFrame({'ret': returns, 'vol': returns**2})
    
    # Lagged correlation
    corrs = []
    lags = range(1, 11)
    for lag in lags:
        # Corr(r_t, sigma^2_{t+k})
        # If negative, negative returns today -> higher vol tomorrow
        c = df['ret'].corr(df['vol'].shift(-lag))
        corrs.append(c)
        
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(lags, corrs, color='#d62728', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title('Leverage Effect: Correlation(Return_t, Volatility_{t+k})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Lag (days)')
    ax.set_ylabel('Correlation')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add explanation
    textstr = "Negative correlation implies\nnegative returns lead to\nhigher future volatility."
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
            
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_leverage_effect.png")
    plt.close()

def plot_rv_vs_vix(df):
    print("Generating RV vs VIX plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Convert to percentage
    rv_30 = df['RV_fwd_30'] * 100
    iv = df['IV'] * 100
    
    # Top panel: Levels
    ax1.plot(rv_30.index, rv_30, label='Realized Volatility (30d)', linewidth=0.8, alpha=0.8, color='#1f77b4')
    ax1.plot(iv.index, iv, label='Implied Volatility (VIX)', linewidth=0.8, alpha=0.8, color='#d62728')
    ax1.set_title('Realized Volatility vs. Implied Volatility (VIX)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Volatility (%)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Spread (Volatility Risk Premium)
    vrp = iv - rv_30
    ax2.plot(vrp.index, vrp, color='black', linewidth=0.6, alpha=0.7)
    ax2.axhline(0, color='red', linestyle='--', linewidth=0.8)
    ax2.fill_between(vrp.index, vrp, 0, where=(vrp>=0), color='green', alpha=0.3, label='Positive Premium')
    ax2.fill_between(vrp.index, vrp, 0, where=(vrp<0), color='red', alpha=0.3, label='Negative Premium')
    ax2.set_title('Volatility Risk Premium (VIX - RV)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Spread (pts)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_rv_vs_vix.png")
    plt.close()

def plot_rv_horizons(df):
    print("Generating RV Horizons plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compare short vs long term
    # Use a subset or rolling mean if too noisy, but raw is honest
    ax.plot(df.index, df['RV_fwd_5'] * 100, label='RV (5-day)', linewidth=0.5, alpha=0.5, color='#1f77b4')
    ax.plot(df.index, df['RV_fwd_30'] * 100, label='RV (30-day)', linewidth=1.2, alpha=0.9, color='#ff7f0e')
    
    ax.set_title('Realized Volatility Term Structure (Short vs Long Term)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Volatility (%)')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_rv_horizons.png")
    plt.close()

def plot_correlation_matrix(df):
    print("Generating Correlation Matrix...")
    # Select relevant columns
    cols = ['RET_SPY', 'IV', 'RV_fwd_5', 'RV_fwd_10', 'RV_fwd_30']
    # Rename for display
    display_names = ['Returns', 'VIX (IV)', 'RV (5d)', 'RV (10d)', 'RV (30d)']
    
    # Drop NaNs for correlation calculation
    corr_df = df[cols].dropna().corr()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                xticklabels=display_names, yticklabels=display_names, 
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    
    ax.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/08_correlation_matrix.png")
    plt.close()

def plot_qq(returns):
    print("Generating QQ plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    stats.probplot(returns, dist="norm", plot=ax)
    
    ax.set_title('Q-Q Plot: Returns vs. Normal Distribution', fontsize=12, fontweight='bold')
    ax.get_lines()[0].set_markerfacecolor('#1f77b4')
    ax.get_lines()[0].set_markeredgecolor('#1f77b4')
    ax.get_lines()[0].set_alpha(0.5)
    ax.get_lines()[0].set_markersize(2)
    ax.get_lines()[1].set_color('#d62728') # Regression line
    ax.get_lines()[1].set_linewidth(2)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/09_qq_plot.png")
    plt.close()

def main():
    df, spy_px, vix_px = load_data()
    returns = df['RET_SPY'].dropna()
    
    plot_market_overview(spy_px, vix_px)
    plot_volatility_clustering(returns)
    plot_distribution_analysis(returns)
    plot_autocorrelation(returns)
    plot_leverage_effect(returns)
    plot_rv_vs_vix(df)
    plot_rv_horizons(df)
    plot_correlation_matrix(df)
    plot_qq(returns)
    
    print(f"\nAll plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
