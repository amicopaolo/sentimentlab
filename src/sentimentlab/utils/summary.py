"""Human-readable summary of OHLCV DataFrames."""

from __future__ import annotations

import pandas as pd

from sentimentlab.formatters.volume import humanize_volume


def summary(df: pd.DataFrame, title: str = "OHLCV Summary") -> str:
    """
    Generate a human-readable summary string for an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Formatted OHLCV DataFrame (should have DatetimeIndex).
    title : str
        Title shown at the top of the summary.

    Returns
    -------
    str
        Multiline summary string.

    Example
    -------
    >>> import sentimentlab as dfm
    >>> print(dfm.summary(clean_df, title="AAPL Daily"))
    """
    lines = [f"{'─' * 40}", f"  {title}", f"{'─' * 40}"]

    # Date range
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
        lines.append(f"  Period    : {df.index[0].date()} → {df.index[-1].date()}")
        lines.append(f"  Rows      : {len(df):,}")

    # Price stats
    for col in ["open", "high", "low", "close", "adj_close"]:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                lines.append(
                    f"  {col.capitalize():<10}: "
                    f"min={s.min():.4f}  max={s.max():.4f}  last={s.iloc[-1]:.4f}"
                )

    # Volume
    if "volume" in df.columns:
        vol = df["volume"].dropna()
        if len(vol) > 0:
            lines.append(
                f"  Volume    : avg={humanize_volume(vol.mean())}  "
                f"total={humanize_volume(vol.sum())}"
            )

    # Missing data
    total_cells = df.shape[0] * df.shape[1]
    missing = df.isnull().sum().sum()
    pct = (missing / total_cells * 100) if total_cells > 0 else 0
    lines.append(f"  NaN cells : {missing:,} ({pct:.1f}%)")

    # Metadata from attrs
    if df.attrs.get("base_currency"):
        lines.append(f"  Currency  : {df.attrs['base_currency']}")
    if df.attrs.get("asset_class"):
        lines.append(f"  Asset     : {df.attrs['asset_class']}")

    lines.append(f"{'─' * 40}")
    return "\n".join(lines)
