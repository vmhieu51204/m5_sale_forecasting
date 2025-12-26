import pandas as pd
import numpy as np

def add_basic_time_features(df):
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df = df.drop(columns=[c for c in ["d", "weekday"] if c in df.columns])
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if col == "id": continue
        df[col] = df[col].fillna("Unknown").astype("category").cat.codes
    if "snap_CA" in df.columns:
        df["snap_CA"] = df["snap_CA"].fillna(0)
    return df

def add_lags_and_rolls(df, value_col, lags, windows, group_col="id"):
    g = df.groupby(group_col)[value_col]
    for lag in lags: df[f"lag_{lag}"] = g.shift(lag)
    for w in windows:
        df[f"roll_mean_{w}"] = g.transform(lambda x: x.shift(1).rolling(w).mean())
        df[f"roll_std_{w}"] = g.transform(lambda x: x.shift(1).rolling(w).std())
    return df.fillna(0)

def process_daily_features(df):
    df = add_basic_time_features(df)
    df = add_lags_and_rolls(df, "sales", lags=[7, 14, 28], windows=[7, 28])
    return df

def process_weekly_agg_features(df):
    df = df.copy() 
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit='D')
    agg_dict = {"sales": "sum", "sell_price": "mean", "snap_CA": "max"}
    for c in ["event_name_1", "item_id", "cat_id", "dept_id"]:
        if c in df.columns: agg_dict[c] = "first"
    
    # Handle missing columns gracefully
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    weekly = df.groupby(["id", "week_start"]).agg(agg_dict).reset_index().rename(columns={"sales": "weekly_sales", "week_start": "date"})
    weekly = add_basic_time_features(weekly)
    return add_lags_and_rolls(weekly, "weekly_sales", lags=[1, 2, 4], windows=[4, 8])

def process_rolling_features(df):
    df = add_basic_time_features(df)
    df["roll_sum_7"] = df.groupby("id")["sales"].transform(lambda x: x.rolling(7).sum())
    df = add_lags_and_rolls(df, "sales", lags=[7, 28], windows=[7, 28])
    for l in [1, 2, 3]: df[f"roll_sum_lag_{l}w"] = df.groupby("id")["roll_sum_7"].shift(l*7)
    return df.fillna(0)