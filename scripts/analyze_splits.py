import pandas as pd
import numpy as np
import yaml
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from volatility_modelling.data.splits import split_by_date, DateSplits

def analyze_splits():
    with open("configs/train.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
        
    df = pd.read_pickle(cfg["paths"]["data_path"])
    target_col = cfg["task"]["target_col"]
    
    splits = DateSplits(
        train_end=cfg["splits"]["train_end"],
        valid_end=cfg["splits"]["valid_end"],
        test_end=cfg["splits"]["test_end"]
    )
    
    train, val, test = split_by_date(df, splits)
    
    print(f"Target Column: {target_col}")
    
    for name, data in [("Train", train), ("Val", val), ("Test", test)]:
        rv = data[target_col]
        print(f"\n{name} Set ({data.index.min().date()} to {data.index.max().date()}):")
        print(f"  Count: {len(rv)}")
        print(f"  Mean: {rv.mean():.4f}")
        print(f"  Std:  {rv.std():.4f}")
        print(f"  Max:  {rv.max():.4f}")
        print(f"  Min:  {rv.min():.4f}")

if __name__ == "__main__":
    analyze_splits()