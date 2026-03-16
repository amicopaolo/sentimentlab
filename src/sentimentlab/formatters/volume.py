"""Volume scaling and human-readable formatting utilities."""

from __future__ import annotations

from typing import Union

import pandas as pd

_SCALE_SUFFIXES = [
    (1_000_000_000, "B"),  # Billion
    (1_000_000, "M"),      # Million
    (1_000, "K"),          # Thousand
]


def scale_volume(
    series: Union[pd.Series, pd.DataFrame],
    factor: float = 1.0,
    unit: str = "shares",
    column: str = "volume",
) -> pd.Series:
    """
    Scale a volume series by a given factor.

    Parameters
    ----------
    series : pd.Series | pd.DataFrame
        Volume data. If DataFrame, uses ``column``.
    factor : float
        Multiplier (e.g., 0.001 to convert to thousands).
    unit : str
        Label stored in the result Series name for traceability.
    column : str
        Column name to use if ``series`` is a DataFrame.

    Returns
    -------
    pd.Series

    Example
    -------
    >>> vol_millions = scale_volume(df, factor=1e-6, unit="M shares")
    """
    if isinstance(series, pd.DataFrame):
        series = series[column]
    result = series * factor
    result.name = f"volume_{unit}"
    return result


def humanize_volume(volume: Union[int, float], decimals: int = 2) -> str:
    """
    Convert a raw volume number into a human-readable string.

    Parameters
    ----------
    volume : int | float
        Raw volume value.
    decimals : int
        Decimal places in the output string.

    Returns
    -------
    str

    Example
    -------
    >>> humanize_volume(1_234_567)
    '1.23M'
    >>> humanize_volume(950)
    '950'
    """
    for threshold, suffix in _SCALE_SUFFIXES:
        if abs(volume) >= threshold:
            scaled = volume / threshold
            return f"{scaled:.{decimals}f}{suffix}"
    return str(int(volume))


def volume_series_stats(series: pd.Series) -> dict[str, float]:
    """
    Return descriptive stats for a volume series useful in market analysis.

    Returns
    -------
    dict with keys: mean, median, std, min, max, total, avg_daily
    """
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "total": float(series.sum()),
        "avg_daily": float(series.mean()),  # alias for clarity
    }
