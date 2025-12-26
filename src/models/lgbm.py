import lightgbm as lgb
import numpy as np
import pandas as pd

def train_lgbm_direct(train_df, val_df, feature_cols, target_cols, params):
    models = {}
    for tgt in target_cols:
        print(f"  Training LGBM for {tgt}...")
        dtrain = lgb.Dataset(train_df[feature_cols], label=train_df[tgt])
        dval = lgb.Dataset(val_df[feature_cols], label=val_df[tgt], reference=dtrain)
        bst = lgb.train(params, dtrain, num_boost_round=300,
                        valid_sets=[dval], callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        models[tgt] = bst
    return models

def predict_lgbm(models, latest_features, feature_cols, start_date, freq="D"):
    preds = []
    ids = latest_features["id"].values
    X_feat = latest_features[feature_cols]
    for tgt_name, model in models.items():
        h_idx = int(tgt_name.split("_")[-1]) 
        y_pred = np.maximum(model.predict(X_feat), 0)
        
        if freq == "W": delta = pd.Timedelta(weeks=h_idx - 1)
        elif freq == "7D": delta = pd.Timedelta(days=(h_idx * 7) - 1)
        else: delta = pd.Timedelta(days=h_idx - 1)
            
        preds.append(pd.DataFrame({
            "id": ids, "date": start_date + delta, "horizon_idx": h_idx,
            "forecast": y_pred, "model_type": "LGBM"
        }))
    return pd.concat(preds, ignore_index=True)