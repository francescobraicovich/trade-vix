import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os
from .base import BaseModel
from .registry import register
from ..training.callbacks import EarlyStopping, ModelCheckpoint
from ..training.scheduler import get_scheduler

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        return self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len - 1]

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional, output_activation):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=bidirectional
        )
        
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        
        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        self.output_activation = output_activation
        self.softplus = nn.Softplus()

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, _ = self.lstm(x)
        
        # Take last time step
        out = out[:, -1, :]
        
        out = self.head(out)
        if self.output_activation == "softplus":
            out = self.softplus(out)
        return out.squeeze(-1)

@register("lstm_rv")
@register("lstm_vix")
class LSTMModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.scaler = StandardScaler()
        self.model = None
        
        # Device selection: cuda > mps > cpu
        device_cfg = cfg["opt"].get("device", "auto")
        if device_cfg == "cpu":
            self.device = torch.device("cpu")
        elif device_cfg == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_cfg == "mps":
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:  # "auto"
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        
        print(f"LSTM using device: {self.device}")
        
    def _build_model(self, input_dim):
        spec = self.cfg["spec"]
        self.model = LSTMNet(
            input_size=input_dim,
            hidden_size=spec["hidden_size"],
            num_layers=spec["num_layers"],
            dropout=spec["dropout"],
            bidirectional=spec["bidirectional"],
            output_activation=spec["output_activation"]
        ).to(self.device)

    def fit(self, train_df, val_df, **kwargs):
        # Data prep
        features_cfg = self.cfg["features"]
        target_cfg = self.cfg["target"]
        opt_cfg = self.cfg["opt"]
        
        input_cols = features_cfg["input_cols"]
        target_col = target_cfg["column"]
        seq_len = features_cfg["seq_len"]
        
        # Fit scaler on train
        X_train = train_df[input_cols].values
        y_train = train_df[target_col].values
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        # Save scaler
        os.makedirs(self.cfg["paths"]["artifacts_dir"] + "/checkpoints", exist_ok=True)
        joblib.dump(self.scaler, f"{self.cfg['paths']['artifacts_dir']}/checkpoints/{self.cfg['model_name']}_scaler.pkl")
        
        # Val data
        X_val = val_df[input_cols].values
        y_val = val_df[target_col].values
        X_val_scaled = self.scaler.transform(X_val)
        
        # Datasets
        train_ds = TimeSeriesDataset(X_train_scaled, y_train, seq_len)
        val_ds = TimeSeriesDataset(X_val_scaled, y_val, seq_len)
        
        train_loader = DataLoader(train_ds, batch_size=opt_cfg["batch_size"], shuffle=True, num_workers=opt_cfg["num_workers"])
        val_loader = DataLoader(val_ds, batch_size=opt_cfg["batch_size"], shuffle=False, num_workers=opt_cfg["num_workers"])
        
        # Build model
        self._build_model(len(input_cols))
        
        # Optimizer & Loss
        optimizer = optim.AdamW(self.model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
        criterion = nn.MSELoss()
        scheduler = get_scheduler(optimizer, opt_cfg["scheduler"], opt_cfg["epochs"])
        
        # Callbacks
        ckpt_path = f"{self.cfg['paths']['artifacts_dir']}/checkpoints/{self.cfg['model_name']}_best.pt"
        early_stopping = EarlyStopping(patience=opt_cfg["early_stop_patience"], mode='min')
        model_checkpoint = ModelCheckpoint(ckpt_path, monitor='val_rmse', mode='min', save_best_only=True)
        
        # Loop
        for epoch in range(opt_cfg["epochs"]):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                if opt_cfg["grad_clip_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), opt_cfg["grad_clip_norm"])
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            train_loss /= len(train_ds)
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    pred = self.model(X_batch)
                    loss = criterion(pred, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
                    val_preds.append(pred.cpu().numpy())
                    val_targets.append(y_batch.cpu().numpy())
            
            val_loss /= len(val_ds)
            val_preds = np.concatenate(val_preds)
            val_targets = np.concatenate(val_targets)
            val_rmse = np.sqrt(np.mean((val_preds - val_targets)**2))
            
            if scheduler:
                scheduler.step()
                
            # Callbacks
            stop = early_stopping(val_rmse)
            model_checkpoint(self.model, val_rmse)
            
            print(f"Epoch {epoch+1}/{opt_cfg['epochs']} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val RMSE: {val_rmse:.6f}")
            
            if stop:
                print("Early stopping triggered")
                break
                
        # Load best model
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        return self

    def predict(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        # df: input dataframe containing features. 
        # We need to create sequences.
        # The output should be aligned with the end of the sequence.
        
        features_cfg = self.cfg["features"]
        input_cols = features_cfg["input_cols"]
        seq_len = features_cfg["seq_len"]
        
        X = df[input_cols].values
        X_scaled = self.scaler.transform(X)
        
        # We need to predict for all possible windows
        # If len(df) < seq_len, can't predict
        if len(df) < seq_len:
            return pd.Series(index=df.index)
            
        # Create dataset for prediction
        # We use a dummy target
        ds = TimeSeriesDataset(X_scaled, np.zeros(len(X)), seq_len)
        loader = DataLoader(ds, batch_size=self.cfg["opt"]["batch_size"], shuffle=False, num_workers=self.cfg["opt"]["num_workers"])
        
        self.model.eval()
        preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                pred = self.model(X_batch)
                preds.append(pred.cpu().numpy())
                
        preds = np.concatenate(preds)
        
        # Align predictions to the index of the last step in the window
        # TimeSeriesDataset yields X[idx : idx + seq_len], so the last timestamp is idx + seq_len - 1
        # The first prediction corresponds to index seq_len - 1
        
        pred_index = df.index[seq_len - 1:]
        return pd.Series(preds, index=pred_index)
