"""
CSV loader and merger for sentiment + yfinance price data.

Expected CSV formats
--------------------
Sentiment CSV (output di FinBERT aggregato per giorno):
    Columns: [date_col, sentiment_col]
    Esempio: Day, daily_sentiment

Finance CSV (output di yf.download(...).to_csv()):
    Riga 0 : header colonne  (Price, AdjClose, Close, ...)
    Riga 1 : nome ticker     (NVDA, NVDA, ...)   ← riga da saltare
    Riga 2+: dati effettivi
    Oppure formato flat senza multi-header (auto-rilevato).
"""

from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd

HORIZONS_DEFAULT = [1, 3, 5, 10, 15, 20, 30, 40]


def load_sentiment_csv(
    path: str,
    date_col: str = "Day",
    sentiment_col: str = "daily_sentiment",
) -> pd.DataFrame:
    """
    Load and normalize a daily sentiment CSV file.

    Parameters
    ----------
    path : str
        Path to the sentiment CSV.
    date_col : str
        Name of the date column (default 'Day').
    sentiment_col : str
        Name of the sentiment score column (default 'daily_sentiment').

    Returns
    -------
    pd.DataFrame
        Columns: ['date', 'sentiment'] with DatetimeIndex-compatible dates.

    Example
    -------
    >>> from sentimentlab.analysis import load_sentiment_csv
    >>> sent = load_sentiment_csv("gdelt_events_90d_nvidia_daily_sentiment.csv")
    """
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.rename(columns={date_col: "date", sentiment_col: "sentiment"})
    if "date" not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in {path}. Available: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "sentiment"]]


def load_finance_csv(
    path: str,
    price_col: str = "Close",
    skip_ticker_row: bool | None = None,
) -> pd.DataFrame:
    """
    Load a yfinance-exported CSV into a normalized DataFrame.

    Handles both the multi-level header format produced by
    ``yf.download(...).to_csv()`` (which has a ticker name row) and
    plain single-header CSVs.

    Parameters
    ----------
    path : str
        Path to the finance CSV.
    price_col : str
        Price column to use for returns (default 'Close').
    skip_ticker_row : bool | None
        If True, skip row index 1 (ticker name row from yfinance multi-header).
        If None (default), auto-detects by checking if row 1 is all non-numeric.

    Returns
    -------
    pd.DataFrame
        Columns: ['Date', 'AdjClose', 'Close', 'High', 'Low', 'Open', 'Volume']
        (subset may vary depending on available columns).

    Example
    -------
    >>> from sentimentlab.analysis import load_finance_csv
    >>> prices = load_finance_csv("yfinance_nvda_90d.csv")
    """
    raw = pd.read_csv(path, header=0)

    # Auto-detect yfinance multi-header (second row = ticker names, all strings)
    if skip_ticker_row is None:
        try:
            second_row = raw.iloc[0]
            # If the second row (index 0 of data) is all non-numeric strings → ticker row
            all_str = all(
                not _is_numeric_like(str(v)) for v in second_row.values if str(v).strip()
            )
            skip_ticker_row = all_str
        except Exception:
            skip_ticker_row = False

    if skip_ticker_row:
        raw = raw.iloc[1:].copy()  # drop ticker row

    # Normalize column names
    raw.columns = [str(c).strip() for c in raw.columns]

    # Rename first column to Date if unnamed
    if raw.columns[0].startswith("Unnamed") or raw.columns[0] == "":
        raw = raw.rename(columns={raw.columns[0]: "Date"})

    # Try to find Date column
    date_candidates = [c for c in raw.columns if c.lower() in ("date", "datetime", "timestamp", "price")]
    if not date_candidates:
        raise ValueError(f"No date column found. Columns: {list(raw.columns)}")
    date_col_name = date_candidates[0]
    raw = raw.rename(columns={date_col_name: "Date"})

    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"])

    # Cast numeric columns
    numeric_cols = [c for c in raw.columns if c != "Date"]
    for col in numeric_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.sort_values("Date").reset_index(drop=True)
    return raw


def merge_sentiment_finance(
    sentiment: pd.DataFrame,
    finance: pd.DataFrame,
    price_col: str = "Close",
    horizons: list[int] = HORIZONS_DEFAULT,
    how: str = "inner",
) -> pd.DataFrame:
    """
    Merge sentiment and finance DataFrames on date, compute forward returns.

    The merge pairs sentiment at day t with returns starting from t+k
    (using ``pct_change(k).shift(-k)``) to avoid look-ahead bias.

    Parameters
    ----------
    sentiment : pd.DataFrame
        Output of :func:`load_sentiment_csv`. Must have columns ['date', 'sentiment'].
    finance : pd.DataFrame
        Output of :func:`load_finance_csv`. Must have column 'Date' and ``price_col``.
    price_col : str
        Price column for return computation (default 'Close').
    horizons : list[int]
        Forward-return horizons in days (default [1, 3, 5, 10, 15, 20, 30, 40]).
    how : str
        Merge strategy: 'inner' (default), 'left', 'outer'.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with columns:
        ['date', 'sentiment', 'Date', price columns..., 'ret_1d', 'ret_3d', ...]
        Also includes 'df_nz' mask via `.attrs['nz_mask']`.

    Example
    -------
    >>> from sentimentlab.analysis import load_sentiment_csv, load_finance_csv, merge_sentiment_finance
    >>> sent = load_sentiment_csv("sentiment.csv")
    >>> prices = load_finance_csv("prices.csv")
    >>> df = merge_sentiment_finance(sent, prices)
    """
    if price_col not in finance.columns:
        available = [c for c in finance.columns if c != "Date"]
        raise ValueError(
            f"Column '{price_col}' not found in finance DataFrame. "
            f"Available: {available}"
        )

    fin = finance.copy()
    for k in horizons:
        fin[f"ret_{k}d"] = fin[price_col].pct_change(k).shift(-k)

    df = pd.merge(sentiment, fin, left_on="date", right_on="Date", how=how)
    df = df.sort_values("date").reset_index(drop=True)

    # Store nz mask for convenience
    df.attrs["nz_mask"] = df["sentiment"] != 0
    df.attrs["horizons"] = horizons
    df.attrs["price_col"] = price_col

    return df


# ── Internal helpers ──────────────────────────────────────────────────────────

def _is_numeric_like(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False
