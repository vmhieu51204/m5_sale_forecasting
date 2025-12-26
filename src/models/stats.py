import pandas as pd
from typing import Dict
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

# Suppress warnings for cleaner output
warnings.simplefilter("ignore")

def forecast_stats_models(
    df: pd.DataFrame,
    freq_str: str,
    target_col: str,
    horizon: int,
    start_date: pd.Timestamp,
    seasonal_period: int = 7
) -> Dict[str, pd.DataFrame]:
    """
    Runs Regular ARIMA (1,1,1), SARIMAX, and ExpSmoothing.
    """
    arima_res = []
    sarimax_res = [] 
    es_res = []
    
    # Iterate through each series
    for sid, sdf in tqdm(df.groupby("id"), desc=f"Stats Models ({freq_str})"):
        # Prepare Univariate Series
        ts = sdf.set_index("date")[target_col].asfreq(freq_str).fillna(0)
        
        # 1. ARIMA (Non-seasonal, fixed order)
        try:
            mod_a = ARIMA(ts, order=(1,1,1)).fit()
            fc_a = mod_a.forecast(horizon)
        except:
            fc_a = pd.Series([ts.iloc[-1]] * horizon)
            
        # 2. SARIMAX (Seasonal)
        try:
            mod_s = SARIMAX(
                ts, 
                order=(1, 1, 1), 
                seasonal_order=(0, 1, 1, seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            fc_s = mod_s.forecast(horizon)
        except:
            fc_s = pd.Series([ts.iloc[-1]] * horizon)

        # 3. Exponential Smoothing (Holt-Winters)
        try:
            mod_e = ExponentialSmoothing(
                ts, 
                trend="add", 
                seasonal="add", 
                seasonal_periods=seasonal_period
            ).fit()
            fc_e = mod_e.forecast(horizon)
        except:
            fc_e = pd.Series([ts.iloc[-1]] * horizon)
            
        # Store Results
        # Ensure forecasts are iterable/list-like
        fc_a = fc_a.values if hasattr(fc_a, "values") else fc_a
        fc_s = fc_s.values if hasattr(fc_s, "values") else fc_s
        fc_e = fc_e.values if hasattr(fc_e, "values") else fc_e

        for h in range(horizon):
            val_a = fc_a[h]
            val_s = fc_s[h]
            val_e = fc_e[h]
            
            # Calculate Forecast Date
            if freq_str.startswith("W"):
                d = start_date + pd.Timedelta(weeks=h)
            elif freq_str == "7D":
                d = start_date + pd.Timedelta(days=(h * 7)) 
            else: # Daily
                d = start_date + pd.Timedelta(days=h)
                
            base = {"id": sid, "date": d, "horizon_idx": h+1}
            
            arima_res.append({**base, "forecast": max(0, val_a), "model_type": "ARIMA"})
            sarimax_res.append({**base, "forecast": max(0, val_s), "model_type": "SARIMAX"})
            es_res.append({**base, "forecast": max(0, val_e), "model_type": "ExpSmoothing"})
            
    return {
        "ARIMA": pd.DataFrame(arima_res),
        "SARIMAX": pd.DataFrame(sarimax_res),
        "ExpSmoothing": pd.DataFrame(es_res)
    }