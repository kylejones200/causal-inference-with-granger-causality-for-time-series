"""Shared project utilities."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import signalplot
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    if not Path(config_path).exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def get_output_dir(config: dict, key: str = "figures_dir") -> Path:
    return Path(config.get("output", {}).get(key, "images"))


def ensure_output_dir(config: dict, key: str = "figures_dir") -> Path:
    out = get_output_dir(config, key)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_plot(path: str | Path, *, close: bool = True) -> None:
    signalplot.save(str(path))
    if close:
        plt.close()


def load_time_series(
    data_path: str | Path,
    date_col: str | None = None,
    value_col: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
    return df


def create_forecast_plot(
    actual: pd.Series,
    forecast: pd.Series,
    title: str = "Forecast",
    output_path: Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actual, label="Actual", color="#2b2b2b")
    ax.plot(forecast, label="Forecast", color=signalplot.ACCENT, linestyle="--")
    ax.set_title(title)
    ax.legend(frameon=False)
    signalplot.tidy_axes(ax)
    if output_path:
        save_plot(output_path)
