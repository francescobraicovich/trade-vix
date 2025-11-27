import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the backtest function
from run_final_pipeline import run_strategies_and_backtest

if __name__ == "__main__":
    print("Running backtest only...")
    run_strategies_and_backtest()
