M5_ROOT = "/kaggle/input/m5-forecasting-accuracy" # Update this path for your local environment
STORE_ID = "CA_1"
DEPT_ID = "FOODS_2"
N_SERIES = 10
YEARS = 2
FORECAST_HORIZON_DAYS = 28

LGB_PARAMS_DAILY = {
    "boosting_type": "gbdt",
    "objective": "tweedie",
    "metric": "rmse",
    "verbose": -1,
    "random_state": 42,
    "force_col_wise": True,
    "learning_rate": 0.06868914676964558,
    "num_leaves": 43,
    "feature_fraction": 0.931668430789315,
    "min_data_in_leaf": 17,
    "tweedie_variance_power": 1.3573314664632754,
}

LSTM_PARAMS_DAILY = {
    "hidden": 64,
    "layers": 2,
    "dropout": 0.1,
    "lr": 0.005,
}

LGB_PARAMS_WEEKLY_AGG = {
    "boosting_type": "gbdt",
    "objective": "tweedie",
    "metric": "rmse",
    "verbose": -1,
    "random_state": 42,
    "force_col_wise": True,
    "learning_rate": 0.045280803658806164,
    "num_leaves": 61,
    "feature_fraction": 0.821116140056114,
    "min_data_in_leaf": 46,
    "tweedie_variance_power": 1.4368176832402675,
}

LSTM_PARAMS_WEEKLY_AGG = {
    "hidden": 64,
    "layers": 2,
    "dropout": 0.1,
    "lr": 0.005,
}

LGB_PARAMS_WEEKLY_ROLL = {
    "boosting_type": "gbdt",
    "objective": "tweedie",
    "metric": "rmse",
    "verbose": -1,
    "random_state": 42,
    "force_col_wise": True,
    "learning_rate": 0.09300550787843843,
    "num_leaves": 62,
    "feature_fraction": 0.9538319704882288,
    "min_data_in_leaf": 32,
    "tweedie_variance_power": 1.3311258880221715,
}

LSTM_PARAMS_WEEKLY_ROLL = {
    "hidden": 64,
    "layers": 2,
    "dropout": 0.1,
    "lr": 0.005,
}