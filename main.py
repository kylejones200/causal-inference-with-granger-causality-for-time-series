"""Granger Causality testing and multivariate forecasting using leading indicators."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.evaluator import Evaluator
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from src import ensure_output_dir, load_config

logger = logging.getLogger(__name__)
def fit_multivariate_regression(
    df: pd.DataFrame, target_col: str, predictor_cols: list[str], lag: int = 1
) -> tuple[sm.OLS, pd.Series]:
    """
    Fit multivariate regression model with lagged predictors.
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with all series
    target_col : str
        Target series name
    predictor_cols : list[str]
        List of predictor series names
    lag : int
        Lag to use for predictors

    Returns:
    --------
    tuple
        (fitted_model, predictions)
    """
    X_data = {}
    for col in predictor_cols:
        X_data[f"{col}_lag{lag}"] = df[col].shift(lag)
    X = pd.DataFrame(X_data, index=df.index)
    aligned = pd.concat([df[[target_col]], X], axis=1).dropna()
    y = aligned[target_col]
    X = aligned[X.columns]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    predictions = pd.Series(model.predict(X), index=aligned.index)
    return (model, predictions)


def forecast_multivariate(
    model: sm.OLS,
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: list[str],
    lag: int,
    forecast_horizon: int,
) -> pd.Series:
    """
    Generate forecasts using multivariate regression.
    Parameters:
    -----------
    model : sm.OLS
        Fitted regression model
    df : pd.DataFrame
        Historical data
    target_col : str
        Target series name
    predictor_cols : list[str]
        Predictor series names
    lag : int
        Lag used in model
    forecast_horizon : int
        Number of steps to forecast

    Returns:
    --------
    pd.Series
        Forecast series with datetime index
    """
    last_values = {}
    for col in predictor_cols:
        last_values[f"{col}_lag{lag}"] = df[col].iloc[-lag]
    last_date = df.index[-1]
    freq = pd.infer_freq(df.index) or "D"
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq=freq
    )
    forecasts = []
    feature_names = []
    if "const" in model.params.index:
        feature_names.append("const")
    for col in predictor_cols:
        feature_names.append(f"{col}_lag{lag}")
    for _ in range(forecast_horizon):
        X_forecast_data = {}
        if "const" in model.params.index:
            X_forecast_data["const"] = [1.0]
        for col in predictor_cols:
            X_forecast_data[f"{col}_lag{lag}"] = [last_values[f"{col}_lag{lag}"]]
        X_forecast = pd.DataFrame(X_forecast_data, index=[0])
        X_forecast = X_forecast.reindex(columns=model.params.index, fill_value=0)
        forecast = model.predict(X_forecast).iloc[0]
        forecasts.append(forecast)
    return pd.Series(forecasts, index=forecast_dates)


def load_multivariate_data(config: dict, script_dir: Path) -> pd.DataFrame:
    """
    Load multivariate time series data.
    Supports:
    - Single CSV with multiple value columns
    - Multiple CSV files (one per series)
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    script_dir : Path
        Script directory for path resolution

    Returns:
    --------
    pd.DataFrame
        DataFrame with datetime index and value columns
    """
    data_config = config["data"]
    repo_root = script_dir.parent
    if "input_file" in data_config:
        first_file = repo_root / data_config["input_file"]
        if "value_columns" in data_config:
            df_full = pd.read_csv(first_file, encoding="utf-8")
            date_col = data_config.get("date_column", "date")
            df_full[date_col] = pd.to_datetime(df_full[date_col], errors="coerce")
            df_full = df_full.dropna(subset=[date_col])
            df_full = df_full.set_index(date_col).sort_index()
            value_cols = data_config["value_columns"]
            series_names = data_config.get("series_names", value_cols)
            df = df_full[value_cols].rename(columns=dict(zip(value_cols, series_names)))
        elif "input_files" in data_config:
            from src import load_time_series

            series_names = data_config.get(
                "series_names", [f"series{i + 1}" for i in range(len(data_config["input_files"]))]
            )
            dfs = []
            for i, file_path in enumerate(data_config["input_files"]):
                file_path = repo_root / file_path
                loaded = load_time_series(
                    str(file_path),
                    date_col=data_config.get("date_col", "date"),
                    value_col=data_config.get("value_col", "value"),
                )
                value_col = data_config.get("value_col", "value")
                series = (
                    loaded[value_col]
                    if value_col in loaded.columns
                    else loaded.iloc[:, 0]
                )
                dfs.append(pd.DataFrame({series_names[i]: series}))
            df = pd.concat(dfs, axis=1)
        else:
            from src import load_time_series

            loaded = load_time_series(
                str(first_file),
                date_col=data_config.get("date_col", "date"),
                value_col=data_config.get("value_col", "value"),
            )
            value_col = data_config.get("value_col", "value")
            first_series = (
                loaded[value_col]
                if value_col in loaded.columns
                else loaded.iloc[:, 0]
            )
            series_names = data_config.get("series_names", ["target", "predictor"])
            target_name = series_names[0]
            predictor_name = series_names[1] if len(series_names) > 1 else "predictor"
            df = pd.DataFrame({target_name: first_series})
            df[predictor_name] = df[target_name].shift(1)
            df = df.dropna()
        return df.dropna()
    raise ValueError("Must specify either 'input_file' or 'input_files' in data config")


def test_granger_causality(
    df: pd.DataFrame, target_col: str, predictor_col: str, max_lag: int = 5, verbose: bool = True
) -> dict:
    """
    Test if predictor_col Granger-causes target_col.
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with both series
    target_col : str
        Target series name
    predictor_col : str
        Predictor series name
    max_lag : int
        Maximum lag to test
    verbose : bool
        Whether to print detailed results

    Returns:
    --------
    dict
        Results dictionary with p-values and causality decision
    """
    test_data = df[[target_col, predictor_col]].dropna()
    if len(test_data) < max_lag + 10:
        raise ValueError(f"Insufficient data: need at least {max_lag + 10} observations")
    if verbose:
        logger.info(f"\nTesting if '{predictor_col}' Granger-causes '{target_col}':")
    gc_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=verbose)
    p_values = {}
    min_p = 1.0
    best_lag = None
    for lag in range(1, max_lag + 1):
        if lag in gc_result:
            p_value = gc_result[lag][0]["ssr_ftest"][1]
            p_values[lag] = p_value
            if p_value < min_p:
                min_p = p_value
                best_lag = lag
    is_causal = min_p < 0.05
    results = {
        "p_values": p_values,
        "min_p_value": min_p,
        "best_lag": best_lag,
        "is_causal": is_causal,
        "interpretation": f"{predictor_col} {('DOES' if is_causal else 'DOES NOT')} Granger-cause {target_col}",
    }
    if verbose:
        logger.info("\nSummary:")
        logger.info(f"  Minimum p-value: {min_p:.4f} (lag {best_lag})")
        logger.info(f"  Causality: {results['interpretation']}")
        if is_causal:
            logger.info(
                f"  → Use {predictor_col} as leading indicator for forecasting {target_col}"
            )
    return results


def test_stationarity(series: pd.Series, name: str) -> bool:
    """
    Test if time series is stationary using Augmented Dickey-Fuller test.
    Parameters:
    -----------
    series : pd.Series
        Time series to test
    name : str
        Series name for reporting

    Returns:
    --------
    bool
        True if stationary (p < 0.05)
    """
    result = adfuller(series.dropna())
    p_value = result[1]
    is_stationary = p_value < 0.05
    logger.info(
        f"  {name}: p-value = {p_value:.4f} {('(stationary)' if is_stationary else '(non-stationary)')}"
    )
    return is_stationary


def _load_and_validate(config: dict, script_dir: Path) -> tuple:
    """Load data and resolve target/predictor columns."""
    df = load_multivariate_data(config, script_dir)
    logger.info(f"Loaded {len(df)} observations with {len(df.columns)} series")
    names = config["data"].get("series_names", list(df.columns))
    if len(names) >= 2:
        for name in names[:2]:
            if name not in df.columns:
                names = list(df.columns)
                break
    if len(names) < 2:
        raise ValueError("Need at least 2 series for Granger causality testing")
    return df, names[0], names[1]


def _analyse_causality(config: dict, df, target_col: str, predictor_col: str) -> tuple:
    """ADF stationarity check + Granger causality test."""
    logger.info("Testing stationarity (ADF):")
    ok_t = test_stationarity(df[target_col], target_col)
    ok_p = test_stationarity(df[predictor_col], predictor_col)
    if not (ok_t and ok_p):
        logger.warning("Non-stationary series detected - consider differencing first.")
    gc = test_granger_causality(
        df, target_col=target_col, predictor_col=predictor_col,
        max_lag=config.get("granger", {}).get("max_lag", 5), verbose=True,
    )
    corr = df[[target_col, predictor_col]].corr().iloc[0, 1]
    logger.info(f"Correlation {target_col} / {predictor_col}: {corr:.4f}")
    return gc, corr


def _fit_and_evaluate(config: dict, df, gc: dict,
                      target_col: str, predictor_col: str, output_dir: Path) -> None:
    """If causal: fit regression, evaluate on test set, generate forecast + plot."""
    if not gc["is_causal"]:
        logger.info(f"No Granger causality found for {predictor_col} to {target_col}.")
        return
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    lag   = gc["best_lag"]
    split = int(len(df) * (1 - config["evaluation"].get("test_size", 0.2)))
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    model, train_pred = fit_multivariate_regression(
        train_df, target_col=target_col, predictor_cols=[predictor_col], lag=lag
    )
    logger.info(model.summary())
    _, test_pred = fit_multivariate_regression(
        test_df, target_col=target_col, predictor_cols=[predictor_col], lag=lag
    )
    actual = test_df[target_col].iloc[lag:]
    pred   = test_pred[test_pred.index.isin(actual.index)]
    actual = actual[actual.index.isin(pred.index)]
    if len(pred) == 0:
        logger.warning("Insufficient test data for evaluation.")
        return
    rmse = np.sqrt(mean_squared_error(actual, pred))
    logger.info(f"Test  RMSE={rmse:.4f}  MAE={mean_absolute_error(actual, pred):.4f}  R2={r2_score(actual, pred):.4f}")

    horizon  = config["evaluation"].get("forecast_horizon", len(test_df))
    forecast = forecast_multivariate(
        model, train_df, target_col=target_col, predictor_cols=[predictor_col],
        lag=lag, forecast_horizon=horizon,
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_df.index, train_df[target_col], "k-", label=f"{target_col} (Train)", alpha=0.7)
    ax.plot(test_df.index,  test_df[target_col],  "g-", label=f"{target_col} (Test)",  alpha=0.7)
    if len(train_pred): ax.plot(train_pred.index, train_pred,  "b--", label="Train Pred", alpha=0.7)
    if len(pred):       ax.plot(pred.index,       pred.values, "r--", label="Test Pred",  alpha=0.7)
    ax.plot(forecast.index, forecast.values, "m--", label="Forecast", linewidth=2)
    ax.set(xlabel="Date", ylabel="Value",
           title=f"Granger Causality Forecast: {target_col} from {predictor_col}")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    plot_path = output_dir / config["output"].get("plot_file", "granger_forecast.png")
    fig.savefig(plot_path, dpi=config["output"].get("dpi", 300), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved: {plot_path}")
    import pandas as pd
    csv = output_dir / config["output"].get("forecast_file", "granger_forecast.csv")
    pd.DataFrame({"date": forecast.index, "forecast": forecast.values}).to_csv(csv, index=False)
    logger.info(f"Forecast saved: {csv}")


def _save_summary(config: dict, gc: dict, corr: float,
                  target_col: str, predictor_col: str, output_dir: Path) -> None:
    import pandas as pd
    path = output_dir / config["output"].get("summary_file", "granger_summary.csv")
    pd.DataFrame({
        "test":        [f"{predictor_col} to {target_col}"],
        "min_p_value": [gc["min_p_value"]],
        "best_lag":    [gc["best_lag"]],
        "is_causal":   [gc["is_causal"]],
        "correlation": [corr],
    }).to_csv(path, index=False, encoding="utf-8")
    logger.info(f"Summary saved: {path}")


def main() -> None:
    script_dir = Path(__file__).parent
    config     = load_config(script_dir / "config.yaml")
    output_dir = ensure_output_dir(config)
    df, target_col, predictor_col = _load_and_validate(config, script_dir)
    gc, corr = _analyse_causality(config, df, target_col, predictor_col)
    _fit_and_evaluate(config, df, gc, target_col, predictor_col, output_dir)
    _save_summary(config, gc, corr, target_col, predictor_col, output_dir)


if __name__ == "__main__":
    main()
