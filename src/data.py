import pandas as pd
import numpy as np

def load_base_data(m5_root, store_id, dept_id, n_series=20, years=2, min_sale_percent=5.0):
    print(f"Loading data for {store_id} - {dept_id}...")
    sales = pd.read_csv(f"{m5_root}/sales_train_validation.csv")
    calendar = pd.read_csv(f"{m5_root}/calendar.csv")
    prices = pd.read_csv(f"{m5_root}/sell_prices.csv")
    mask = (sales["store_id"] == store_id) & (sales["dept_id"] == dept_id)
    sales = sales[mask].head(n_series).reset_index(drop=True)
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    d_cols = [c for c in sales.columns if c.startswith("d_")]
    if years > 0: d_cols = d_cols[-(years * 365):]
    df = sales[id_cols + d_cols].melt(id_vars=id_cols, value_vars=d_cols, var_name="d", value_name="sales")
    sale_pct = df.groupby("id")["sales"].apply(lambda x: (x > 0).mean() * 100)
    active_ids = sale_pct[sale_pct >= min_sale_percent].index
    df = df[df["id"].isin(active_ids)].reset_index(drop=True)
    df = df.merge(calendar, on="d", how="left")
    df["date"] = pd.to_datetime(df["date"])
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    df["sell_price"] = df.groupby("id")["sell_price"].ffill().fillna(0)
    return df.sort_values(["id", "date"]).reset_index(drop=True)

def split_time_based(df, cutoff_date):
    train = df[df["date"] < cutoff_date].copy()
    test = df[df["date"] >= cutoff_date].copy()
    return train, test

def get_cutoff_date(df, days_back=28):
    return df["date"].max() - pd.Timedelta(days=days_back - 1)

def create_multi_horizon_targets(df, target_col, horizon, unit_step=1, prefix="target_"):
    df = df.copy()
    g = df.groupby("id")[target_col]
    target_names = []
    for h in range(1, horizon + 1):
        col_name = f"{prefix}{h}"
        df[col_name] = g.shift(-h * unit_step)
        target_names.append(col_name)
    return df.dropna(subset=target_names), target_names