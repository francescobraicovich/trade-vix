import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from volatility_modelling.training.train_loop import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_cfg", type=str, required=True)
    parser.add_argument("--model_cfg", type=str, required=True)
    args = parser.parse_args()
    
    run_training(args.train_cfg, args.model_cfg)
