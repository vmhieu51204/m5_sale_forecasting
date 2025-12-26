# src/evaluation.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def standardize_to_weekly(df, freq_type):
    df = df.copy()
    if freq_type == "daily":
        df["week_num"] = ((df["horizon_idx"] - 1) // 7) + 1
        return df.groupby(["id", "week_num"])["forecast"].sum().reset_index()
    elif freq_type in ["weekly_agg", "weekly_roll"]:
        df = df.rename(columns={"horizon_idx": "week_num"})
        return df[["id", "week_num", "forecast"]]
    return pd.DataFrame()

def evaluate_all_models(results_registry, actuals_daily, start_date):
    act_daily = actuals_daily[actuals_daily["date"] >= start_date].copy()
    act_daily["week_num"] = ((act_daily["date"] - start_date).dt.days // 7) + 1
    act_daily = act_daily[act_daily["date"] < start_date + pd.Timedelta(days=28)]
    act_weekly = act_daily.groupby(["id", "week_num"])["sales"].sum().reset_index().rename(columns={"sales": "actual"})
    act_28d = act_weekly.groupby("id")["actual"].sum()
    
    metrics = []
    for method_name, (df, ftype) in results_registry.items():
        if df.empty: continue
        row = {"Method": method_name}
        
        # Daily metrics
        if ftype == "daily":
            merged = df.merge(act_daily, on=["id", "date"], how="inner")
            if not merged.empty:
                row["Daily RMSE"] = np.sqrt(mean_squared_error(merged["sales"], merged["forecast"]))
                row["Daily MAE"] = mean_absolute_error(merged["sales"], merged["forecast"])
        else:
            row["Daily RMSE"] = np.nan
            row["Daily MAE"] = np.nan

        # Weekly & Total metrics
        std_pred = standardize_to_weekly(df, ftype)
        merged_w = act_weekly.merge(std_pred, on=["id", "week_num"], how="inner")
        if not merged_w.empty:
            row["Weekly RMSE"] = np.sqrt(mean_squared_error(merged_w["actual"], merged_w["forecast"]))
            row["Weekly MAE"] = mean_absolute_error(merged_w["actual"], merged_w["forecast"])
            
            grp_pred = merged_w.groupby("id")["forecast"].sum()
            common = act_28d.index.intersection(grp_pred.index)
            row["Total 28D RMSE"] = np.sqrt(mean_squared_error(act_28d[common], grp_pred[common]))
            row["Total 28D MAE"] = mean_absolute_error(act_28d[common], grp_pred[common])
            row["Total 28D Bias"] = (grp_pred[common] - act_28d[common]).mean()
        
        metrics.append(row)
    
    res_df = pd.DataFrame(metrics)
    cols = ["Method", "Weekly RMSE", "Weekly MAE", "Total 28D RMSE", "Total 28D MAE", "Total 28D Bias"]
    if "Daily RMSE" in res_df.columns: 
        cols = ["Method", "Daily RMSE", "Daily MAE"] + cols[1:]
    return res_df[cols].sort_values("Weekly RMSE")

def print_strategy_metrics(forecast_df, actuals_df, start_date, strategy_name, freq_type):
    print(f"\n>>> PERFORMANCE METRICS: {strategy_name}")
    act_filtered = actuals_df[actuals_df["date"] >= start_date].copy()
    
    std_pred = standardize_to_weekly(forecast_df, freq_type)
    
    act_filtered["week_num"] = ((act_filtered["date"] - start_date).dt.days // 7) + 1
    act_filtered = act_filtered[act_filtered["week_num"] <= 4] 
    weekly_act = act_filtered.groupby(["id", "week_num"])["sales"].sum().reset_index()
    merged = weekly_act.merge(std_pred, on=["id", "week_num"], how="inner")
    
    if merged.empty:
        print("  (No overlapping data for metrics calculation)")
        return

    w_rmse = np.sqrt(mean_squared_error(merged["sales"], merged["forecast"]))
    w_mae = mean_absolute_error(merged["sales"], merged["forecast"])
    
    vol_act = weekly_act.groupby("id")["sales"].sum()
    vol_pred = std_pred.groupby("id")["forecast"].sum()
    common = vol_act.index.intersection(vol_pred.index)
    
    vol_rmse = np.sqrt(mean_squared_error(vol_act[common], vol_pred[common]))
    vol_mae = mean_absolute_error(vol_act[common], vol_pred[common])
    vol_bias = (vol_pred[common] - vol_act[common]).mean()
    
    metrics_df = pd.DataFrame({
        "Metric Level": ["Weekly Aggregated", "Total 28-Day Volume"],
        "RMSE": [w_rmse, vol_rmse], "MAE": [w_mae, vol_mae], "Bias": [np.nan, vol_bias]
    })
    print(metrics_df.to_string(index=False, float_format="%.4f"))
    print("-" * 60)