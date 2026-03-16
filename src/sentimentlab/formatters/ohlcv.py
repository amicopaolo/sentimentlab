"""OHLCV (Open/High/Low/Close/Volume) DataFrame formatting utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Standard column name aliases → canonical name
_COLUMN_ALIASES: dict[str, str] = {
    # Open
    "open": "open", "Open": "open", "OPEN": "open", "o": "open",
    # High
    "high": "high", "High": "high", "HIGH": "high", "h": "high",
    # Low
    "low": "low", "Low": "low", "LOW": "low", "l": "low",
    # Close
    "close": "close", "Close": "close", "CLOSE": "close", "c": "close",
    "adj close": "adj_close", "Adj Close": "adj_close",
    # Volume
    "volume": "volume", "Volume": "volume", "VOLUME": "volume", "v": "volume", "vol": "volume",
    # Timestamp
    "date": "timestamp", "Date": "timestamp", "datetime": "timestamp",
    "Datetime": "timestamp", "time": "timestamp", "Time": "timestamp",
    "timestamp": "timestamp", "Timestamp": "timestamp",
}

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]


@dataclass
class OHLCVFormatter:
    """
    Configurable formatter for OHLCV DataFrames.

    Parameters
    ----------
    price_decimals : int
        Decimal places for price columns (default 4).
    volume_decimals : int
        Decimal places for volume column (default 0).
    drop_duplicates : bool
        Remove duplicate timestamps (default True).
    sort_ascending : bool
        Sort rows by timestamp ascending (default True).
    fill_missing : bool
        Forward-fill NaN values in OHLCV columns (default False).
    require_volume : bool
        Raise if volume column is missing (default False).

    Example
    -------
    >>> fmt = OHLCVFormatter(price_decimals=2)
    >>> clean_df = fmt.format(raw_df)
    """

    price_decimals: int = 4
    volume_decimals: int = 0
    drop_duplicates: bool = True
    sort_ascending: bool = True
    fill_missing: bool = False
    require_volume: bool = False
    extra_columns: list[str] = field(default_factory=list)

    def format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a cleaned, normalized OHLCV DataFrame."""
        df = df.copy()
        df = _rename_columns(df)
        df = _set_timestamp_index(df)
        df = _cast_numeric(df, PRICE_COLUMNS + ["volume"])

        if self.require_volume and "volume" not in df.columns:
            raise ValueError("Column 'volume' is required but missing from DataFrame.")

        if self.drop_duplicates:
            df = df[~df.index.duplicated(keep="last")]

        if self.sort_ascending:
            df = df.sort_index(ascending=True)

        if self.fill_missing:
            price_cols = [c for c in PRICE_COLUMNS if c in df.columns]
            df[price_cols] = df[price_cols].ffill()
            if "volume" in df.columns:
                df["volume"] = df["volume"].fillna(0)

        # Round prices
        price_cols = [c for c in PRICE_COLUMNS if c in df.columns]
        df[price_cols] = df[price_cols].round(self.price_decimals)
        if "volume" in df.columns:
            df["volume"] = df["volume"].round(self.volume_decimals)

        # Validate OHLC integrity: high >= max(open, close) and low <= min(open, close)
        _validate_ohlc_integrity(df)

        keep_cols = [c for c in OHLCV_COLUMNS + ["adj_close"] + self.extra_columns if c in df.columns]
        return df[keep_cols]


def format_ohlcv(
    df: pd.DataFrame,
    price_decimals: int = 4,
    volume_decimals: int = 0,
    drop_duplicates: bool = True,
    sort_ascending: bool = True,
    fill_missing: bool = False,
    require_volume: bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper around :class:`OHLCVFormatter`.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame with flexible column names.
    price_decimals : int
        Decimal places for price columns.
    volume_decimals : int
        Decimal places for volume.
    drop_duplicates : bool
        Remove duplicate timestamps.
    sort_ascending : bool
        Sort by timestamp ascending.
    fill_missing : bool
        Forward-fill NaN price values.
    require_volume : bool
        Raise ValueError if volume column is absent.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with DatetimeIndex and canonical column names.

    Example
    -------
    >>> import yfinance as yf
    >>> import sentimentlab as dfm
    >>> raw = yf.download("AAPL", start="2024-01-01", end="2024-06-01")
    >>> clean = dfm.format_ohlcv(raw, price_decimals=2)
    """
    fmt = OHLCVFormatter(
        price_decimals=price_decimals,
        volume_decimals=volume_decimals,
        drop_duplicates=drop_duplicates,
        sort_ascending=sort_ascending,
        fill_missing=fill_missing,
        require_volume=require_volume,
    )
    return fmt.format(df)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename DataFrame columns to canonical lowercase names."""
    rename_map = {col: _COLUMN_ALIASES[col] for col in df.columns if col in _COLUMN_ALIASES}
    return df.rename(columns=rename_map)


def _set_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    """Move 'timestamp' column to DatetimeIndex, or convert existing index."""
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df.index.name = "timestamp"
    else:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index.name = "timestamp"
    return df


def _cast_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce specified columns to float64, replacing non-numeric with NaN."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _validate_ohlc_integrity(df: pd.DataFrame) -> None:
    """Log a warning (not raise) when OHLC relationships are violated."""
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return
    bad_high = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
    bad_low = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
    if bad_high > 0 or bad_low > 0:
        import warnings
        warnings.warn(
            f"OHLC integrity: {bad_high} rows where high < max(open,close), "
            f"{bad_low} rows where low > min(open,close). "
            "Data may contain errors or adjustments.",
            stacklevel=3,
        )
