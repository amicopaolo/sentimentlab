"""
Generate synthetic sample CSV files for sentimentlab tutorials.

Usage
-----
    import sentimentlab as sl
    sl.make_sample_data()
    # → writes sample_sentiment.csv and sample_prices.csv in the current directory

    # Custom output directory
    sl.make_sample_data(output_dir="my_data/")

    # Custom parameters
    sl.make_sample_data(ticker="AAPL", days=120, seed=99)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def make_sample_data(
    ticker: str = "NVDA",
    days: int = 90,
    seed: int = 42,
    output_dir: str = ".",
    sentiment_file: str = "sample_sentiment.csv",
    prices_file: str = "sample_prices.csv",
    verbose: bool = True,
) -> tuple[str, str]:
    """
    Generate two synthetic CSV files that mimic real sentimentlab inputs:

    - **sample_sentiment.csv**: daily sentiment scores in [-1, 1]
      (format: ``Day``, ``daily_sentiment``)
    - **sample_prices.csv**: OHLCV price data with realistic random walk
      (format: ``Date``, ``Open``, ``High``, ``Low``, ``Close``, ``AdjClose``, ``Volume``)

    No internet connection or external data required.

    Parameters
    ----------
    ticker : str
        Ticker label used in the printed summary (default 'NVDA').
    days : int
        Number of trading days to generate (default 90).
    seed : int
        Random seed for reproducibility (default 42).
    output_dir : str
        Directory where CSV files are written (default: current directory).
    sentiment_file : str
        Filename for the sentiment CSV (default 'sample_sentiment.csv').
    prices_file : str
        Filename for the prices CSV (default 'sample_prices.csv').
    verbose : bool
        Print confirmation messages (default True).

    Returns
    -------
    tuple[str, str]
        Absolute paths to (sentiment_csv, prices_csv).

    Example
    -------
    >>> import sentimentlab as sl
    >>> sent_path, prices_path = sl.make_sample_data()
    >>> sent   = sl.load_sentiment_csv(sent_path)
    >>> prices = sl.load_finance_csv(prices_path)
    >>> df     = sl.merge_sentiment_finance(sent, prices)
    >>> print(sl.test_adf_stationarity(df))
    """
    rng = np.random.default_rng(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Sentiment CSV ──────────────────────────────────────────────────────────
    # Business days only (Mon–Fri), mimicking GDELT daily aggregation
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)

    # Simulate FinBERT-style scores: mostly small values, occasional spikes
    base = rng.normal(0, 0.25, size=days)
    spikes = rng.choice([0.0, 0.0, 0.0, 0.6, -0.6], size=days)  # rare extremes
    scores = np.clip(base + spikes, -1.0, 1.0)
    scores[rng.integers(0, days, size=days // 10)] = 0.0  # neutral days

    sent_df = pd.DataFrame({
        "Day": dates.strftime("%Y-%m-%d"),
        "daily_sentiment": np.round(scores, 6),
    })
    sent_path = out / sentiment_file
    sent_df.to_csv(sent_path, index=False)

    # ── Prices CSV ─────────────────────────────────────────────────────────────
    # Random walk with slight downward drift (mimics NVDA bear period in report)
    # Extra 10 days so merge_sentiment_finance has full forward-return coverage
    price_days = days + 45
    price_dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=price_days)

    start_price = 800.0
    daily_returns = rng.normal(-0.001, 0.022, size=price_days)  # slight negative drift
    prices = start_price * np.cumprod(1 + daily_returns)

    open_  = prices * rng.uniform(0.990, 1.005, size=price_days)
    high   = np.maximum(open_, prices) * rng.uniform(1.000, 1.015, size=price_days)
    low    = np.minimum(open_, prices) * rng.uniform(0.985, 1.000, size=price_days)
    volume = rng.integers(20_000_000, 120_000_000, size=price_days).astype(float)

    prices_df = pd.DataFrame({
        "Date":     price_dates.strftime("%Y-%m-%d"),
        "Open":     np.round(open_, 4),
        "High":     np.round(high, 4),
        "Low":      np.round(low, 4),
        "Close":    np.round(prices, 4),
        "AdjClose": np.round(prices * 0.9997, 4),
        "Volume":   volume,
    })
    prices_path = out / prices_file
    prices_df.to_csv(prices_path, index=False)

    if verbose:
        print(f"[sentimentlab] Sample data generated for '{ticker}'")
        print(f"  Sentiment : {sent_path}  ({days} days)")
        print(f"  Prices    : {prices_path}  ({price_days} days)")
        print()
        print("  Quick start:")
        print("    import sentimentlab as sl")
        print(f"    sent   = sl.load_sentiment_csv('{sent_path}')")
        print(f"    prices = sl.load_finance_csv('{prices_path}')")
        print("    df     = sl.merge_sentiment_finance(sent, prices)")
        print("    print(sl.test_pearson_spearman(df))")

    return str(sent_path.resolve()), str(prices_path.resolve())
