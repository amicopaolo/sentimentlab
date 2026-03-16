"""
sentimentlab CLI — run all 9 statistical tests from the command line.

Usage
-----
    sentimentlab --sentiment sentiment.csv --finance prices.csv
    sentimentlab --sentiment sentiment.csv --finance prices.csv --test adf pearson ols
    sentimentlab --sentiment sentiment.csv --finance prices.csv --horizons 1 5 10 20
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

TEST_NAMES = [
    "adf", "pearson", "ols", "leadlag", "granger",
    "event", "ttest", "backtest", "rolling",
]

ALL_TESTS = TEST_NAMES


def _run_tests(args: argparse.Namespace) -> None:
    import sentimentlab as sl

    print(f"\n[sentimentlab v{sl.__version__}]")
    print(f"Loading sentiment : {args.sentiment}")
    print(f"Loading finance   : {args.finance}\n")

    sent = sl.load_sentiment_csv(
        args.sentiment,
        date_col=args.date_col,
        sentiment_col=args.sentiment_col,
    )
    prices = sl.load_finance_csv(args.finance, price_col=args.price_col)
    horizons = list(args.horizons)
    df = sl.merge_sentiment_finance(sent, prices, price_col=args.price_col, horizons=horizons)
    print(f"Merged DataFrame: {len(df)} rows, horizons={horizons}\n")

    tests_to_run = args.test or ALL_TESTS

    def _sep(title: str) -> None:
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"{'─' * 60}")

    if "adf" in tests_to_run:
        _sep("4.1 ADF Stationarity Test")
        print(sl.test_adf_stationarity(df).to_string(index=False))

    if "pearson" in tests_to_run:
        _sep("4.2 Pearson & Spearman Correlations")
        print(sl.test_pearson_spearman(df, horizons=horizons).to_string(index=False))

    if "ols" in tests_to_run:
        _sep("4.3 OLS Regression")
        print(sl.test_ols_regression(df, horizons=horizons).to_string(index=False))

    if "leadlag" in tests_to_run:
        _sep("4.4 Lead-Lag Analysis")
        print(sl.test_lead_lag(df).to_string(index=False))

    if "granger" in tests_to_run:
        _sep("4.5 Granger Causality")
        print(sl.test_granger_causality(df).to_string(index=False))

    if "event" in tests_to_run:
        _sep("4.6 Event-Based Analysis")
        print(sl.test_event_based(df, horizons=horizons).to_string(index=False))

    if "ttest" in tests_to_run:
        _sep("4.7 T-Test HighBull vs HighBear")
        print(sl.test_ttest_bull_vs_bear(df, horizons=horizons).to_string(index=False))

    if "backtest" in tests_to_run:
        _sep("4.8 Backtest Strategies")
        print(sl.test_backtest_strategies(df).to_string(index=False))

    if "rolling" in tests_to_run:
        _sep("4.9 Rolling Correlation")
        _, summary = sl.test_rolling_correlation(df)
        print(summary.to_string(index=False))

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sentimentlab",
        description="Run sentiment vs. market statistical tests.",
    )
    parser.add_argument("--sentiment", required=True, help="Path to sentiment CSV")
    parser.add_argument("--finance", required=True, help="Path to yfinance price CSV")
    parser.add_argument("--date-col", default="Day", dest="date_col",
                        help="Date column name in sentiment CSV (default: Day)")
    parser.add_argument("--sentiment-col", default="daily_sentiment", dest="sentiment_col",
                        help="Sentiment column name (default: daily_sentiment)")
    parser.add_argument("--price-col", default="Close", dest="price_col",
                        help="Price column for returns (default: Close)")
    parser.add_argument("--horizons", nargs="+", type=int,
                        default=[1, 3, 5, 10, 15, 20, 30, 40],
                        help="Forward-return horizons in days")
    parser.add_argument("--test", nargs="+", choices=TEST_NAMES, metavar="TEST",
                        help=f"Tests to run (default: all). Choices: {TEST_NAMES}")
    args = parser.parse_args()

    try:
        _run_tests(args)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
