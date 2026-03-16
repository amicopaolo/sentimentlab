"""
sentimentlab
~~~~~~~~~~~~

A Python library for sentiment-driven financial market analysis.

Provides tools to:
- Load and normalize daily sentiment CSVs (e.g. from FinBERT/GDELT)
- Load yfinance price data and compute forward returns
- Run 9 statistical tests: ADF, Pearson/Spearman, OLS, Lead-Lag,
  Granger causality, Event-Based, Bull-vs-Bear t-test, Backtest, Rolling corr.

Quick start::

    import sentimentlab as sl

    sent   = sl.load_sentiment_csv("sentiment.csv")
    prices = sl.load_finance_csv("prices.csv")
    df     = sl.merge_sentiment_finance(sent, prices)

    print(sl.test_adf_stationarity(df))
    print(sl.test_pearson_spearman(df))
    print(sl.test_backtest_strategies(df))

:copyright: (c) 2026 Paolo Amico
:license: MIT
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("sentimentlab")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0-dev"

__author__ = "Paolo Amico"
__email__ = "paolo.amicopk@gmail.com"

# ── Data loading ──────────────────────────────────────────────────────────────
from sentimentlab.analysis.loader import (
    load_sentiment_csv,
    load_finance_csv,
    merge_sentiment_finance,
)

# ── Statistical tests (sections 4.1–4.9) ─────────────────────────────────────
from sentimentlab.analysis.tests import (
    test_adf_stationarity,
    test_pearson_spearman,
    test_ols_regression,
    test_lead_lag,
    test_granger_causality,
    test_event_based,
    test_ttest_bull_vs_bear,
    test_backtest_strategies,
    test_rolling_correlation,
)

# ── OHLCV formatting ──────────────────────────────────────────────────────────
from sentimentlab.formatters.ohlcv import format_ohlcv, OHLCVFormatter
from sentimentlab.formatters.price import normalize_prices, round_price
from sentimentlab.formatters.volume import scale_volume, humanize_volume
from sentimentlab.formatters.ticker import normalize_ticker, TickerFormatter

# ── Parsers ───────────────────────────────────────────────────────────────────
from sentimentlab.parsers.timestamp import parse_timestamp, align_timestamps
from sentimentlab.parsers.currency import detect_currency, convert_currency_code

# ── Utils ─────────────────────────────────────────────────────────────────────
from sentimentlab.utils.summary import summary
from sentimentlab.utils.validation import validate_ohlcv

__all__ = [
    # Loaders
    "load_sentiment_csv",
    "load_finance_csv",
    "merge_sentiment_finance",
    # Tests
    "test_adf_stationarity",
    "test_pearson_spearman",
    "test_ols_regression",
    "test_lead_lag",
    "test_granger_causality",
    "test_event_based",
    "test_ttest_bull_vs_bear",
    "test_backtest_strategies",
    "test_rolling_correlation",
    # Formatters
    "format_ohlcv",
    "OHLCVFormatter",
    "normalize_prices",
    "round_price",
    "scale_volume",
    "humanize_volume",
    "normalize_ticker",
    "TickerFormatter",
    # Parsers
    "parse_timestamp",
    "align_timestamps",
    "detect_currency",
    "convert_currency_code",
    # Utils
    "summary",
    "validate_ohlcv",
    # Meta
    "__version__",
]
