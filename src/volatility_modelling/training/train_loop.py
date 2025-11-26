import pandas as pd
import numpy as np
import yaml
import os
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ..data.splits import split_by_date, DateSplits
from ..models.registry import build
# Import models to ensure they are registered
from ..models import garch, lstm
from ..utils.plotting import plot_predictions
from ..data.preprocessing import Preprocessor

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def deep_merge(dict1, dict2):
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def run_training(train_cfg_path, model_cfg_path):
    train_cfg = load_config(train_cfg_path)
    model_cfg = load_config(model_cfg_path)
    
    # Merge configs for model (deep merge to preserve nested keys)
    full_cfg = deep_merge(model_cfg, train_cfg)
    
    # Set seeds
    seed = train_cfg["random_seed"]
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # Load data
    data_path = train_cfg["paths"]["data_path"]
    df = pd.read_pickle(data_path)
    df = df.sort_index()

    # Prepare multiple horizons: compute forward and backward realized vol for each horizon
    horizons = train_cfg.get("horizons", [30])
    annualization = train_cfg.get("opt", {}).get("annualization", 252)
    ret_col = train_cfg["task"]["ret_col"]

    for h in horizons:
        # Forward realized vol: uses future returns (t+1 .. t+h)
        fwd_sq = 0
        for i in range(1, h + 1):
            fwd_sq = fwd_sq + df[ret_col].shift(-i).pow(2)
        df[f"RV_fwd_{h}"] = np.sqrt((annualization / h) * fwd_sq)

        # Backward realized vol: past h returns up to t (t-h+1 .. t)
        back_sq = df[ret_col].pow(2).rolling(window=h).sum()
        df[f"RV_back_{h}"] = np.sqrt((annualization / h) * back_sq)

    # Drop rows with NaNs introduced by forward/backward calculations
    df = df.dropna()
    
    # Run training for each horizon separately and save artifacts per-horizon
    base_artifacts = Path(train_cfg["paths"]["artifacts_dir"])
    splits_cfg = train_cfg["splits"]
    splits = DateSplits(
        train_end=splits_cfg["train_end"],
        valid_end=splits_cfg["valid_end"],
        test_end=splits_cfg["test_end"]
    )

    for h in horizons:
        print(f"\n=== Running horizon: {h} days ===")
        # Prepare column names
        target_col = f"RV_fwd_{h}"
        back_col = f"RV_back_{h}"

        # Prepare per-horizon artifacts dir
        artifacts_dir_h = base_artifacts / f"h_{h}"
        artifacts_dir_h.mkdir(parents=True, exist_ok=True)

        # Prepare preprocessing cfg per-horizon (ensure is applied to relevant columns)
        pre_cfg = train_cfg.get("preprocessing", {}).copy()
        log_cols = set(pre_cfg.get("log_transform_cols", []))
        log_cols.update([target_col, back_col, train_cfg["task"]["iv_col"]])
        pre_cfg["log_transform_cols"] = list(log_cols)

        preprocessor = Preprocessor(pre_cfg)

        # Split data for this horizon
        train_df, val_df, test_df = split_by_date(df, splits)

        # Check stationarity before transform (train only)
        preprocessor.check_stationarity(train_df, f"Train (Raw) h={h}")

        preprocessor.fit(train_df)

        train_df_h = preprocessor.transform(train_df)
        val_df_h = preprocessor.transform(val_df)
        test_df_h = preprocessor.transform(test_df)

        # Update full cfg for this horizon
        full_cfg_h = deep_merge(model_cfg, train_cfg)
        # ensure artifacts path and target column are set per-horizon
        full_cfg_h.setdefault("paths", {})["artifacts_dir"] = str(artifacts_dir_h)
        full_cfg_h.setdefault("task", {})["target_col"] = target_col

        # For GARCH, set horizon in spec if present
        if "spec" in full_cfg_h:
            full_cfg_h["spec"]["horizon_days"] = h

        # For LSTM RV model, ensure input column uses back_col
        if full_cfg_h.get("model_name") == "lstm_rv":
            full_cfg_h.setdefault("features", {})["input_cols"] = [back_col]
            full_cfg_h.setdefault("target", {})["column"] = target_col
        
        # For LSTM VIX model, ensure target column is set correctly
        if full_cfg_h.get("model_name") == "lstm_vix":
            full_cfg_h.setdefault("target", {})["column"] = target_col

        # Build model for this horizon
        model = build(full_cfg_h["model_name"], full_cfg_h)

        # Fit
        if full_cfg_h["model_name"] == "garch":
            model.fit(train_df_h[ret_col])
        else:
            model.fit(train_df_h, val_df_h)

        # Predict and evaluate
        predictions = {}
        metrics = {}

        for split_name, split_df in [("train", train_df_h), ("val", val_df_h), ("test", test_df_h)]:
            if split_df.empty:
                continue

            if full_cfg_h["model_name"] == "garch":
                # GARCH predicts from raw returns and outputs in original volatility scale
                preds = model.predict(df[ret_col], split_df.index)
            else:
                preds = model.predict(split_df)

            common_idx = preds.index.intersection(split_df.index)
            y_true = split_df.loc[common_idx, target_col]
            y_pred = preds.loc[common_idx]

            # Inverse transform if needed
            # Note: GARCH already predicts in original scale, so skip inverse transform for it
            y_true = preprocessor.inverse_transform_target(y_true, target_col)
            if full_cfg_h["model_name"] != "garch":
                y_pred = preprocessor.inverse_transform_target(y_pred, target_col)

            valid_mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]

            if len(y_true) > 0:
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                metrics[f"{split_name}_rmse"] = float(rmse)
                metrics[f"{split_name}_mae"] = float(mae)
                predictions[split_name] = {"y_true": y_true, "y_pred": y_pred}

        # Save artifacts per-horizon
        model_name = full_cfg_h["model_name"]
        with open(artifacts_dir_h / f"{model_name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        for split_name, data in predictions.items():
            pd.to_pickle(data, artifacts_dir_h / f"{model_name}_{split_name}_preds.pkl")
            if split_name in ["val", "test"]:
                plot_predictions(data["y_true"], data["y_pred"], f"{model_name} - h{h} - {split_name}", artifacts_dir_h / f"{model_name}_h{h}_{split_name}_plot.png")

        print(f"Finished training {model_name} for horizon {h}")
        print(json.dumps(metrics, indent=4))

