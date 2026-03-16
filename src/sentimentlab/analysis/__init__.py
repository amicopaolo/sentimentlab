"""
Analysis package — sentiment-vs-market statistical tests.

Based on the NVDA/GLD analysis methodology from the research report.
Each function corresponds to one test section (4.1–4.9).
"""

from sentimentlab.analysis.loader import load_sentiment_csv, load_finance_csv, merge_sentiment_finance
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

__all__ = [
    "load_sentiment_csv",
    "load_finance_csv",
    "merge_sentiment_finance",
    "test_adf_stationarity",
    "test_pearson_spearman",
    "test_ols_regression",
    "test_lead_lag",
    "test_granger_causality",
    "test_event_based",
    "test_ttest_bull_vs_bear",
    "test_backtest_strategies",
    "test_rolling_correlation",
]
