# src/models/lstm.py
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, df, feature_cols, target_cols, group_col="id", seq_len=28):
        self.samples = []
        for _, sub in df.groupby(group_col):
            sub = sub.sort_values("date")
            X = sub[feature_cols].values.astype(np.float32)
            Y = sub[target_cols].values.astype(np.float32)
            
            if len(sub) <= seq_len: continue

            for t in range(len(sub) - seq_len):
                self.samples.append((X[t : t+seq_len], Y[t+seq_len-1]))
                
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

def scale_dfs(train_df, val_df, f_cols, t_cols):
    sc_f, sc_t = StandardScaler(), StandardScaler()
    tr_s, va_s = train_df.copy(), val_df.copy()
    
    tr_s[f_cols] = sc_f.fit_transform(train_df[f_cols])
    tr_s[t_cols] = sc_t.fit_transform(train_df[t_cols])
    
    va_s[f_cols] = sc_f.transform(val_df[f_cols])
    va_s[t_cols] = sc_t.transform(val_df[t_cols])
    
    return tr_s, va_s, sc_t, sc_f

def run_lstm_pipeline(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    full_df: pd.DataFrame,
    feature_cols: List[str], 
    target_cols: List[str],
    start_date: pd.Timestamp,
    params: Dict,
    freq: str = "D"
) -> pd.DataFrame:
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    SEQ_LEN = 28
    
    hidden = params.get("hidden", 64)
    layers = params.get("layers", 2)
    drp = params.get("dropout", 0.1)
    lr = params.get("lr", 0.005)
    
    print(f"  [LSTM] Scaling Data...")
    tr_s, va_s, sc_t, sc_f = scale_dfs(train_df, val_df, feature_cols, target_cols)

    ds_train = TimeSeriesDataset(tr_s, feature_cols, target_cols, seq_len=SEQ_LEN)
    
    if len(ds_train) == 0:
        print("Not enough data for LSTM.")
        return pd.DataFrame()

    loader = DataLoader(ds_train, batch_size=64, shuffle=True)
    
    model = LSTMModel(len(feature_cols), len(target_cols), hidden, layers, drp).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    
    print(f"  [LSTM] Training ({freq}) | Params: {params}")
    for ep in range(5): 
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
        print(f"  Epoch {ep+1} Loss: {np.mean(losses):.4f}")
        
    model.eval()
    preds = []
    
    full_s = full_df.copy()
    full_s[feature_cols] = sc_f.transform(full_df[feature_cols].fillna(0))
    
    last_seqs = full_s.sort_values("date").groupby("id").tail(SEQ_LEN)
    
    with torch.no_grad():
        for sid, sub in last_seqs.groupby("id"):
            if len(sub) < SEQ_LEN: continue
            seq = torch.tensor(sub[feature_cols].values, dtype=torch.float32).unsqueeze(0).to(device)
            out_scaled = model(seq).cpu().numpy()[0]
            out = sc_t.inverse_transform(out_scaled.reshape(1, -1))[0]
            out = np.maximum(out, 0) 
            
            for i, val in enumerate(out):
                h_idx = i + 1
                if freq == "W": d = start_date + pd.Timedelta(weeks=i)
                elif freq == "7D": d = start_date + pd.Timedelta(days=(h_idx*7)-1)
                else: d = start_date + pd.Timedelta(days=i)
                
                preds.append({
                    "id": sid, 
                    "date": d, 
                    "horizon_idx": h_idx, 
                    "forecast": val, 
                    "model_type": "LSTM"
                })
                
    return pd.DataFrame(preds)