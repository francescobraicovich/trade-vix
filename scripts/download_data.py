#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from volatility_modelling.data.features import DataBuildConfig, build_and_save_streams

def parse_args():
    p = argparse.ArgumentParser(description="Download SPY/VIX and build RV targets + combined dataset.")
    p.add_argument("--config", type=Path, default=Path("configs/data.yaml"), help="Path to data.yaml config.")
    p.add_argument("--start", type=str, default=None, help="Override start date (YYYY-MM-DD)." )
    p.add_argument("--end", type=str, default=None, help="Override end date (YYYY-MM-DD)." )
    p.add_argument("--spy", type=str, default=None, help="Override SPY proxy ticker (default from config)." )
    p.add_argument("--vix", type=str, default=None, help="Override VIX ticker (default from config)." )
    p.add_argument("--horizons", type=str, default=None, help="Comma-separated list of horizons (e.g., '2,5,10,30')." )
    p.add_argument("--out", type=Path, default=None, help="Output base directory (default from config)." )
    return p.parse_args()

def main():
    args = parse_args()
    cfg_dict = yaml.safe_load(args.config.read_text())
    data_cfg = cfg_dict.get("data", {})
    cfg = DataBuildConfig(**data_cfg)

    # Overrides
    if args.start: cfg.start_date = args.start
    if args.end: cfg.end_date = args.end
    if args.spy: cfg.s_and_p_proxy = args.spy
    if args.vix: cfg.vix_ticker = args.vix
    if args.horizons: cfg.horizons = [int(h.strip()) for h in args.horizons.split(",")]
    if args.out: cfg.output_dir = args.out

    print(f"Building data with horizons: {cfg.horizons}")
    paths = build_and_save_streams(cfg)
    print("\nArtifacts written:")
    for k, v in paths.items():
        print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()
