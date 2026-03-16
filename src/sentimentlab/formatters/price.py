"""Price normalization and rounding utilities."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Union

import pandas as pd

# Decimal places by asset class convention
_ASSET_DECIMALS: dict[str, int] = {
    "equity": 2,
    "fx": 5,
    "crypto": 8,
    "commodity": 3,
    "bond": 4,
    "index": 2,
}


def round_price(
    value: Union[float, int],
    decimals: int = 2,
    method: str = "half_up",
) -> float:
    """
    Round a single price value.

    Parameters
    ----------
    value : float | int
        Price to round.
    decimals : int
        Number of decimal places.
    method : str
        Rounding method: 'half_up' (default) or 'standard' (Python built-in).

    Returns
    -------
    float

    Example
    -------
    >>> round_price(1.2345678, decimals=4)
    1.2346
    """
    if method == "half_up":
        quantize_str = "1." + "0" * decimals if decimals > 0 else "1"
        return float(Decimal(str(value)).quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP))
    return round(float(value), decimals)


def normalize_prices(
    df: pd.DataFrame,
    base_currency: str = "USD",
    decimals: int | None = None,
    asset_class: str = "equity",
    price_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Normalize price columns in a DataFrame to a standard format.

    Currently handles:
    - Rounding to asset-class-appropriate decimals
    - Removing negative prices (replace with NaN)
    - Removing zero prices (replace with NaN) for non-volume columns

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price columns.
    base_currency : str
        Target currency label (stored in df.attrs for traceability).
    decimals : int | None
        Override decimal places. If None, uses asset-class default.
    asset_class : str
        One of: 'equity', 'fx', 'crypto', 'commodity', 'bond', 'index'.
    price_columns : list[str] | None
        Columns to normalize. Defaults to ['open','high','low','close','adj_close'].

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized price columns and metadata in attrs.

    Example
    -------
    >>> df = dfm.normalize_prices(raw_df, asset_class="crypto", decimals=6)
    """
    df = df.copy()
    default_cols = ["open", "high", "low", "close", "adj_close"]
    cols = price_columns or [c for c in default_cols if c in df.columns]

    dp = decimals if decimals is not None else _ASSET_DECIMALS.get(asset_class, 2)

    for col in cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        # Remove negatives (data errors)
        series = series.where(series >= 0, other=float("nan"))
        # Remove zeros in price columns (not volume)
        series = series.where(series != 0, other=float("nan"))
        df[col] = series.round(dp)

    # Store metadata
    df.attrs["base_currency"] = base_currency
    df.attrs["asset_class"] = asset_class
    df.attrs["price_decimals"] = dp

    return df


def pct_change_series(
    series: pd.Series,
    periods: int = 1,
    annualize: bool = False,
    trading_days: int = 252,
) -> pd.Series:
    """
    Compute percentage change for a price series.

    Parameters
    ----------
    series : pd.Series
        Price series (e.g., close prices).
    periods : int
        Number of periods for the change calculation.
    annualize : bool
        If True, annualize the result assuming ``trading_days`` per year.
    trading_days : int
        Number of trading days in a year (default 252).

    Returns
    -------
    pd.Series

    Example
    -------
    >>> returns = pct_change_series(df["close"], annualize=True)
    """
    pct = series.pct_change(periods=periods)
    if annualize:
        pct = pct * (trading_days / periods)
    return pct
