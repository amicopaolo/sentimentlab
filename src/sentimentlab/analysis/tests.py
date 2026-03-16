"""
Statistical tests for sentiment vs. market-return analysis.

Each function corresponds to one section of the analysis report:

    4.1  test_adf_stationarity       — ADF unit-root test
    4.2  test_pearson_spearman       — Pearson & Spearman correlations
    4.3  test_ols_regression         — OLS simple regression
    4.4  test_lead_lag               — Cross-correlation at lags −15…+15
    4.5  test_granger_causality      — Granger causality (bidir.)
    4.6  test_event_based            — One-sample t-test per regime × horizon
    4.7  test_ttest_bull_vs_bear     — Independent t-test HighBull vs HighBear
    4.8  test_backtest_strategies    — Long / Long-Short / Buy&Hold backtest
    4.9  test_rolling_correlation    — Rolling Pearson r over a sliding window
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

HORIZONS_DEFAULT = [1, 3, 5, 10, 15, 20, 30, 40]


# ─────────────────────────────────────────────────────────────────────────────
# 4.1  ADF Stationarity Test
# ─────────────────────────────────────────────────────────────────────────────

def test_adf_stationarity(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment",
    return_col: str = "ret_1d",
) -> pd.DataFrame:
    """
    Augmented Dickey-Fuller test for stationarity on sentiment and return series.

    A significant result (p < 0.05) means the series is stationary — a
    necessary condition for valid correlation and regression analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame from :func:`merge_sentiment_finance`.
    sentiment_col : str
        Name of the sentiment column (default 'sentiment').
    return_col : str
        Return column to test (default 'ret_1d').

    Returns
    -------
    pd.DataFrame
        Columns: ['Series', 'ADF_stat', 'p_value', 'is_stationary']

    Example
    -------
    >>> from sentimentlab.analysis import test_adf_stationarity
    >>> result = test_adf_stationarity(df)
    >>> print(result)
    """
    results = []
    for col in [sentiment_col, return_col]:
        series = df[col].dropna()
        if len(series) < 10:
            warnings.warn(f"Column '{col}' has fewer than 10 non-NaN values; ADF skipped.")
            continue
        adf_stat, p_val, *_ = adfuller(series)
        results.append({
            "Series": col,
            "ADF_stat": round(float(adf_stat), 4),
            "p_value": round(float(p_val), 4),
            "is_stationary": p_val < 0.05,
        })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 4.2  Pearson & Spearman Correlations
# ─────────────────────────────────────────────────────────────────────────────

def test_pearson_spearman(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment",
    horizons: list[int] = HORIZONS_DEFAULT,
    include_nonzero: bool = True,
) -> pd.DataFrame:
    """
    Pearson and Spearman correlations between sentiment and forward returns
    at multiple horizons, on the full sample and on non-zero sentiment only.

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame with columns 'sentiment' and 'ret_{k}d'.
    sentiment_col : str
        Sentiment column name.
    horizons : list[int]
        Forward-return horizons in days.
    include_nonzero : bool
        If True, also compute on subset where sentiment != 0 (default True).

    Returns
    -------
    pd.DataFrame
        Columns: ['Orizzonte', 'Campione', 'N', 'Pearson_r', 'Pearson_p',
                  'Spearman_r', 'Spearman_p']

    Example
    -------
    >>> corr = test_pearson_spearman(df)
    >>> corr[corr["Pearson_p"] < 0.05]
    """
    df_nz = df[df[sentiment_col] != 0].copy()
    subsets = [("Tutti", df)]
    if include_nonzero:
        subsets.append(("Sent!=0", df_nz))

    results = []
    for k in horizons:
        col = f"ret_{k}d"
        if col not in df.columns:
            continue
        for name, subset_base in subsets:
            subset = subset_base[[sentiment_col, col]].dropna()
            if len(subset) < 5:
                continue
            pr, pp = stats.pearsonr(subset[sentiment_col], subset[col])
            sr, sp = stats.spearmanr(subset[sentiment_col], subset[col])
            results.append({
                "Orizzonte": f"{k}d",
                "Campione": name,
                "N": len(subset),
                "Pearson_r": round(float(pr), 4),
                "Pearson_p": round(float(pp), 4),
                "Spearman_r": round(float(sr), 4),
                "Spearman_p": round(float(sp), 4),
            })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 4.3  OLS Regression
# ─────────────────────────────────────────────────────────────────────────────

def test_ols_regression(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment",
    horizons: list[int] = HORIZONS_DEFAULT,
    include_nonzero: bool = True,
) -> pd.DataFrame:
    """
    OLS simple regression: return_kd ~ β·sentiment + α, for each horizon.

    Returns Beta (slope), p-value for Beta, and R².

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame.
    sentiment_col : str
        Sentiment column.
    horizons : list[int]
        Forward-return horizons.
    include_nonzero : bool
        Include non-zero-sentiment subset.

    Returns
    -------
    pd.DataFrame
        Columns: ['Orizzonte', 'Campione', 'N', 'Beta', 'p_beta', 'R2']

    Example
    -------
    >>> reg = test_ols_regression(df)
    >>> reg[reg["p_beta"] < 0.05]
    """
    df_nz = df[df[sentiment_col] != 0].copy()
    subsets = [("Tutti", df)]
    if include_nonzero:
        subsets.append(("Sent!=0", df_nz))

    results = []
    for k in horizons:
        col = f"ret_{k}d"
        if col not in df.columns:
            continue
        for name, subset_base in subsets:
            subset = subset_base[[sentiment_col, col]].dropna()
            if len(subset) < 5:
                continue
            sl, it, rv, pv, se = stats.linregress(subset[sentiment_col], subset[col])
            results.append({
                "Orizzonte": f"{k}d",
                "Campione": name,
                "N": len(subset),
                "Beta": round(float(sl), 6),
                "p_beta": round(float(pv), 4),
                "R2": round(float(rv ** 2), 4),
            })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 4.4  Lead-Lag Analysis
# ─────────────────────────────────────────────────────────────────────────────

def test_lead_lag(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment",
    return_col: str = "ret_1d",
    lag_range: tuple[int, int] = (-15, 15),
    min_n: int = 5,
) -> pd.DataFrame:
    """
    Cross-correlation at lags from ``lag_range[0]`` to ``lag_range[1]``.

    Negative lag k means sentiment is shifted k days into the future
    (i.e., return leads sentiment). Positive lag k means sentiment leads return.

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame.
    sentiment_col : str
        Sentiment column.
    return_col : str
        Return column (default 'ret_1d').
    lag_range : tuple[int, int]
        (min_lag, max_lag) inclusive (default (-15, 15)).
    min_n : int
        Minimum observations per lag to include (default 5).

    Returns
    -------
    pd.DataFrame
        Columns: ['Lag', 'Pearson_r', 'p_value', 'N']

    Example
    -------
    >>> lag = test_lead_lag(df)
    >>> lag[lag["p_value"] < 0.05]
    """
    results = []
    for lag in range(lag_range[0], lag_range[1] + 1):
        temp = df[[sentiment_col, return_col]].copy()
        temp["sent_lag"] = temp[sentiment_col].shift(lag)
        temp = temp[["sent_lag", return_col]].dropna()
        temp_nz = temp[temp["sent_lag"] != 0]
        if len(temp_nz) < min_n:
            continue
        r, p = stats.pearsonr(temp_nz["sent_lag"], temp_nz[return_col])
        results.append({
            "Lag": lag,
            "Pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4),
            "N": len(temp_nz),
        })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 4.5  Granger Causality
# ─────────────────────────────────────────────────────────────────────────────

def test_granger_causality(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment",
    return_col: str = "ret_1d",
    maxlag: int = 5,
) -> pd.DataFrame:
    """
    Granger causality test in both directions:
    - Sentiment → Return
    - Return → Sentiment

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame.
    sentiment_col : str
        Sentiment column.
    return_col : str
        Return column.
    maxlag : int
        Maximum lag to test (default 5).

    Returns
    -------
    pd.DataFrame
        Columns: ['Direzione', 'Lag', 'F_stat', 'p_value']

    Example
    -------
    >>> gc = test_granger_causality(df)
    >>> gc[gc["p_value"] < 0.05]
    """
    gc_df = df[[sentiment_col, return_col]].dropna()
    results = []

    directions = [
        (return_col, sentiment_col, "Sentiment → Rendimento"),
        (sentiment_col, return_col, "Rendimento → Sentiment"),
    ]

    for y_col, x_col, label in directions:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gc = grangercausalitytests(gc_df[[y_col, x_col]], maxlag=maxlag, verbose=False)
        for lag_k in range(1, maxlag + 1):
            f_stat = gc[lag_k][0]["ssr_ftest"][0]
            f_p = gc[lag_k][0]["ssr_ftest"][1]
            results.append({
                "Direzione": label,
                "Lag": lag_k,
                "F_stat": round(float(f_stat), 4),
                "p_value": round(float(f_p), 4),
            })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 4.6  Event-Based Analysis
# ─────────────────────────────────────────────────────────────────────────────

def test_event_based(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment",
    horizons: list[int] = HORIZONS_DEFAULT,
    q_low: float = 0.30,
    q_high: float = 0.70,
    min_n: int = 3,
) -> pd.DataFrame:
    """
    One-sample t-test for each sentiment regime × horizon.

    Regimes are defined on the non-zero sentiment subset:
    - HighBull: sentiment >= q_high quantile
    - HighBear: sentiment <= q_low quantile
    - Neutro: between the two quantiles

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame.
    sentiment_col : str
        Sentiment column.
    horizons : list[int]
        Forward-return horizons.
    q_low : float
        Lower quantile threshold (default 0.30).
    q_high : float
        Upper quantile threshold (default 0.70).
    min_n : int
        Minimum observations per cell.

    Returns
    -------
    pd.DataFrame
        Columns: ['Regime', 'Orizzonte', 'N', 'Ret_medio_%', 't_stat', 'p_value']

    Example
    -------
    >>> ev = test_event_based(df)
    >>> ev[(ev["Regime"] == "HighBear") & (ev["p_value"] < 0.05)]
    """
    df_nz = df[df[sentiment_col] != 0].copy()
    q30 = df_nz[sentiment_col].quantile(q_low)
    q70 = df_nz[sentiment_col].quantile(q_high)

    df_nz["regime"] = "Neutro"
    df_nz.loc[df_nz[sentiment_col] >= q70, "regime"] = "HighBull"
    df_nz.loc[df_nz[sentiment_col] <= q30, "regime"] = "HighBear"

    results = []
    for regime in ["HighBull", "HighBear", "Neutro"]:
        sub = df_nz[df_nz["regime"] == regime]
        for k in horizons:
            col = f"ret_{k}d"
            if col not in df.columns:
                continue
            vals = sub[col].dropna()
            if len(vals) < min_n:
                continue
            t_stat, t_p = stats.ttest_1samp(vals, 0)
            results.append({
                "Regime": regime,
                "Orizzonte": f"{k}d",
                "N": len(vals),
                "Ret_medio_%": round(float(vals.mean() * 100), 3),
                "t_stat": round(float(t_stat), 3),
                "p_value": round(float(t_p), 4),
            })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 4.7  T-Test HighBull vs HighBear
# ─────────────────────────────────────────────────────────────────────────────

def test_ttest_bull_vs_bear(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment",
    horizons: list[int] = HORIZONS_DEFAULT,
    q_low: float = 0.30,
    q_high: float = 0.70,
    min_n: int = 3,
) -> pd.DataFrame:
    """
    Independent two-sample Welch t-test: HighBull returns vs HighBear returns
    at each horizon.

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame.
    sentiment_col : str
        Sentiment column.
    horizons : list[int]
        Forward-return horizons.
    q_low : float
        Bear quantile (default 0.30).
    q_high : float
        Bull quantile (default 0.70).
    min_n : int
        Minimum observations per group.

    Returns
    -------
    pd.DataFrame
        Columns: ['Orizzonte', 'Bull_mean_%', 'Bear_mean_%', 'Diff_%',
                  't_stat', 'p_value']

    Example
    -------
    >>> tt = test_ttest_bull_vs_bear(df)
    >>> tt[tt["p_value"] < 0.05]
    """
    df_nz = df[df[sentiment_col] != 0].copy()
    q30 = df_nz[sentiment_col].quantile(q_low)
    q70 = df_nz[sentiment_col].quantile(q_high)

    df_nz["regime"] = "Neutro"
    df_nz.loc[df_nz[sentiment_col] >= q70, "regime"] = "HighBull"
    df_nz.loc[df_nz[sentiment_col] <= q30, "regime"] = "HighBear"

    results = []
    for k in horizons:
        col = f"ret_{k}d"
        if col not in df.columns:
            continue
        bv = df_nz[df_nz["regime"] == "HighBull"][col].dropna()
        brv = df_nz[df_nz["regime"] == "HighBear"][col].dropna()
        if len(bv) < min_n or len(brv) < min_n:
            continue
        t, p = stats.ttest_ind(bv, brv, equal_var=False)
        results.append({
            "Orizzonte": f"{k}d",
            "Bull_mean_%": round(float(bv.mean() * 100), 3),
            "Bear_mean_%": round(float(brv.mean() * 100), 3),
            "Diff_%": round(float((bv.mean() - brv.mean()) * 100), 3),
            "t_stat": round(float(t), 3),
            "p_value": round(float(p), 4),
        })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 4.8  Backtest Strategies
# ─────────────────────────────────────────────────────────────────────────────

def test_backtest_strategies(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment",
    return_col: str = "ret_1d",
    q_low: float = 0.30,
    q_high: float = 0.70,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Compare three daily trading strategies:

    - **Long Sent>Q70**: long only when sentiment >= q_high quantile
    - **Long/Short**: long when sent >= q_high, short when sent <= q_low
    - **Buy & Hold**: hold every day

    Metrics: total return %, annualized Sharpe ratio, max drawdown %.

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame.
    sentiment_col : str
        Sentiment column.
    return_col : str
        Daily return column (default 'ret_1d').
    q_low : float
        Bear quantile for Short signal.
    q_high : float
        Bull quantile for Long signal.
    trading_days : int
        Annualization factor (default 252).

    Returns
    -------
    pd.DataFrame
        Columns: ['Strategia', 'Ret_tot_%', 'Sharpe', 'MaxDD_%', 'N_trades']

    Example
    -------
    >>> bt = test_backtest_strategies(df)
    >>> print(bt.to_string())
    """
    df_strat = df.copy().dropna(subset=[return_col])
    df_nz = df_strat[df_strat[sentiment_col] != 0]

    q30 = df_nz[sentiment_col].quantile(q_low)
    q70 = df_nz[sentiment_col].quantile(q_high)

    # Signal 1: long when bullish
    df_strat["sig1"] = 0
    df_strat.loc[
        (df_strat[sentiment_col] >= q70) & (df_strat[sentiment_col] != 0), "sig1"
    ] = 1

    # Signal 2: long/short
    df_strat["sig2"] = 0
    df_strat.loc[
        (df_strat[sentiment_col] >= q70) & (df_strat[sentiment_col] != 0), "sig2"
    ] = 1
    df_strat.loc[
        (df_strat[sentiment_col] <= q30) & (df_strat[sentiment_col] != 0), "sig2"
    ] = -1

    df_strat["ret_s1"] = df_strat["sig1"] * df_strat[return_col]
    df_strat["ret_s2"] = df_strat["sig2"] * df_strat[return_col]
    df_strat["cum_s1"] = (1 + df_strat["ret_s1"]).cumprod()
    df_strat["cum_s2"] = (1 + df_strat["ret_s2"]).cumprod()
    df_strat["cum_bh"] = (1 + df_strat[return_col]).cumprod()

    def _calc_metrics(rets: pd.Series, cum: pd.Series) -> tuple[float, float, float]:
        total = float(cum.iloc[-1] - 1)
        sharpe = (
            float(rets.mean() / rets.std() * np.sqrt(trading_days))
            if rets.std() > 0 else 0.0
        )
        dd = float((cum / cum.cummax() - 1).min())
        return total, sharpe, dd

    s1_tot, s1_sh, s1_dd = _calc_metrics(df_strat["ret_s1"], df_strat["cum_s1"])
    s2_tot, s2_sh, s2_dd = _calc_metrics(df_strat["ret_s2"], df_strat["cum_s2"])
    bh_tot, bh_sh, bh_dd = _calc_metrics(df_strat[return_col], df_strat["cum_bh"])

    n_s1 = int((df_strat["sig1"] == 1).sum())
    n_s2 = int((df_strat["sig2"] != 0).sum())

    return pd.DataFrame({
        "Strategia": ["Long Sent>Q70", "Long/Short", "Buy & Hold"],
        "Ret_tot_%": [round(s1_tot * 100, 2), round(s2_tot * 100, 2), round(bh_tot * 100, 2)],
        "Sharpe": [round(s1_sh, 3), round(s2_sh, 3), round(bh_sh, 3)],
        "MaxDD_%": [round(s1_dd * 100, 2), round(s2_dd * 100, 2), round(bh_dd * 100, 2)],
        "N_trades": [n_s1, n_s2, len(df_strat)],
    })


# ─────────────────────────────────────────────────────────────────────────────
# 4.9  Rolling Correlation
# ─────────────────────────────────────────────────────────────────────────────

def test_rolling_correlation(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment",
    return_col: str = "ret_5d",
    window: int = 20,
    min_nz: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling Pearson correlation between sentiment and a forward-return column
    over a sliding window, computed on non-zero sentiment observations.

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame.
    sentiment_col : str
        Sentiment column.
    return_col : str
        Forward-return column (default 'ret_5d').
    window : int
        Rolling window size in rows (default 20).
    min_nz : int
        Minimum non-zero observations per window (default 5).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - ``rolling_df``: row-by-row results with columns
          ['date', 'rolling_r', 'p_value', 'n_nz']
        - ``summary_df``: descriptive statistics DataFrame

    Example
    -------
    >>> rolling, summary = test_rolling_correlation(df)
    >>> print(summary)
    """
    date_col = "date" if "date" in df.columns else df.columns[0]
    df_roll = df[[date_col, sentiment_col, return_col]].dropna().copy()
    df_roll = df_roll.sort_values(date_col).reset_index(drop=True)

    rolling_corrs = []
    for i in range(window, len(df_roll)):
        chunk = df_roll.iloc[i - window: i]
        chunk_nz = chunk[chunk[sentiment_col] != 0]
        if len(chunk_nz) < min_nz:
            continue
        r, p = stats.pearsonr(chunk_nz[sentiment_col], chunk_nz[return_col])
        rolling_corrs.append({
            "date": df_roll.iloc[i][date_col],
            "rolling_r": float(r),
            "p_value": float(p),
            "n_nz": len(chunk_nz),
        })

    rolling_df = pd.DataFrame(rolling_corrs)
    if rolling_df.empty:
        return rolling_df, pd.DataFrame()

    n_sig = int((rolling_df["p_value"] < 0.05).sum())
    n_total = len(rolling_df)
    summary_df = pd.DataFrame({
        "Metrica": [
            "N finestre",
            "Rolling r min",
            "Rolling r max",
            "Rolling r media",
            "Finestre con p < 0.05",
        ],
        "Valore": [
            str(n_total),
            str(round(rolling_df["rolling_r"].min(), 3)),
            str(round(rolling_df["rolling_r"].max(), 3)),
            str(round(rolling_df["rolling_r"].mean(), 3)),
            f"{n_sig}/{n_total} ({round(n_sig / n_total * 100, 0):.0f}%)",
        ],
    })
    return rolling_df, summary_df
