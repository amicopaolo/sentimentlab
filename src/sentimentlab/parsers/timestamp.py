"""Timestamp parsing and alignment utilities for financial time series."""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd
import pytz
from dateutil import parser as dateutil_parser

# Common financial market timezone abbreviations
_MARKET_TIMEZONES: dict[str, str] = {
    "NYSE":   "America/New_York",
    "NASDAQ": "America/New_York",
    "LSE":    "Europe/London",
    "XETRA":  "Europe/Berlin",
    "MTA":    "Europe/Rome",      # Borsa Italiana
    "TSE":    "Asia/Tokyo",
    "HKEX":   "Asia/Hong_Kong",
    "SSE":    "Asia/Shanghai",
    "ASX":    "Australia/Sydney",
    "UTC":    "UTC",
}


def parse_timestamp(
    value: Union[str, int, float, pd.Timestamp],
    tz: str = "UTC",
    unit: str = "s",
) -> pd.Timestamp:
    """
    Parse a timestamp from various input formats into a tz-aware pd.Timestamp.

    Parameters
    ----------
    value : str | int | float | pd.Timestamp
        Input timestamp. Can be ISO string, Unix epoch (int/float), or Timestamp.
    tz : str
        Target timezone. Use market codes like 'NYSE' or IANA names like 'Europe/Rome'.
    unit : str
        Unit for numeric epochs: 's' (seconds, default) or 'ms' (milliseconds).

    Returns
    -------
    pd.Timestamp (tz-aware)

    Example
    -------
    >>> parse_timestamp("2024-01-15 09:30:00", tz="NYSE")
    Timestamp('2024-01-15 09:30:00-0500', tz='America/New_York')
    >>> parse_timestamp(1705312200, tz="UTC")
    Timestamp('2024-01-15 09:30:00+0000', tz='UTC')
    """
    tz_name = _MARKET_TIMEZONES.get(tz, tz)

    if isinstance(value, (int, float)):
        if unit == "ms":
            value = value / 1000
        ts = pd.Timestamp.fromtimestamp(value, tz=pytz.utc)
    elif isinstance(value, str):
        try:
            ts = pd.Timestamp(value)
        except ValueError:
            dt = dateutil_parser.parse(value)
            ts = pd.Timestamp(dt)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
    elif isinstance(value, pd.Timestamp):
        ts = value
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
    else:
        raise TypeError(f"Unsupported timestamp type: {type(value)}")

    return ts.tz_convert(tz_name)


def align_timestamps(
    df: pd.DataFrame,
    freq: str = "D",
    tz: str = "UTC",
    method: str = "ffill",
    market: str | None = None,
) -> pd.DataFrame:
    """
    Resample and align a financial DataFrame to a regular frequency.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex.
    freq : str
        Pandas offset alias: 'D' (daily), 'H' (hourly), 'T' (minute), etc.
    tz : str
        Target timezone for the output index.
    method : str
        Gap-fill method: 'ffill', 'bfill', or 'none'.
    market : str | None
        If provided, uses market timezone override (e.g., 'NYSE').

    Returns
    -------
    pd.DataFrame

    Example
    -------
    >>> daily = align_timestamps(intraday_df, freq="D", market="NYSE")
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")

    target_tz = _MARKET_TIMEZONES.get(market, market) if market else tz

    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")

    df = df.tz_convert(target_tz)

    # For OHLCV resample, use specific aggregations if columns exist
    agg_map: dict[str, str] = {}
    if "open" in df.columns:
        agg_map["open"] = "first"
    if "high" in df.columns:
        agg_map["high"] = "max"
    if "low" in df.columns:
        agg_map["low"] = "min"
    if "close" in df.columns:
        agg_map["close"] = "last"
    if "adj_close" in df.columns:
        agg_map["adj_close"] = "last"
    if "volume" in df.columns:
        agg_map["volume"] = "sum"

    if agg_map:
        # Only resample recognized OHLCV columns; pass through others with last
        other_cols = {c: "last" for c in df.columns if c not in agg_map}
        agg_map.update(other_cols)
        df = df.resample(freq).agg(agg_map)
    else:
        df = df.resample(freq).last()

    if method == "ffill":
        df = df.ffill()
    elif method == "bfill":
        df = df.bfill()

    return df
