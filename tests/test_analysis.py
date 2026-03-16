"""
Pytest test suite for sentimentlab.analysis (tests 4.1–4.9).

Uses synthetic data that mimics the NVDA/GDELT CSV format.
"""

import numpy as np
import pandas as pd
import pytest

import sentimentlab as sl
from sentimentlab.analysis.loader import load_sentiment_csv, load_finance_csv, merge_sentiment_finance


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_sentiment(tmp_path) -> str:
    """Write a synthetic sentiment CSV and return its path."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    scores = rng.uniform(-1, 1, size=90)
    scores[::5] = 0.0  # some neutral days
    df = pd.DataFrame({"Day": dates.strftime("%Y-%m-%d"), "daily_sentiment": scores})
    path = tmp_path / "sentiment.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def synthetic_finance(tmp_path) -> str:
    """Write a synthetic yfinance-style CSV and return its path."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=95, freq="D")
    price = 100.0
    closes = [price]
    for _ in range(94):
        price = price * (1 + rng.normal(0, 0.015))
        closes.append(price)
    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Open": np.array(closes) * rng.uniform(0.99, 1.01, 95),
        "High": np.array(closes) * rng.uniform(1.00, 1.02, 95),
        "Low": np.array(closes) * rng.uniform(0.98, 1.00, 95),
        "Close": closes,
        "AdjClose": np.array(closes) * 0.99,
        "Volume": rng.integers(1_000_000, 50_000_000, size=95),
    })
    path = tmp_path / "finance.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def merged_df(synthetic_sentiment, synthetic_finance) -> pd.DataFrame:
    sent = sl.load_sentiment_csv(synthetic_sentiment)
    prices = sl.load_finance_csv(synthetic_finance)
    return sl.merge_sentiment_finance(sent, prices, horizons=[1, 3, 5, 10, 15, 20, 30, 40])


# ─────────────────────────────────────────────────────────────────────────────
# Loader tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLoaders:
    def test_load_sentiment_columns(self, synthetic_sentiment):
        df = sl.load_sentiment_csv(synthetic_sentiment)
        assert set(df.columns) == {"date", "sentiment"}
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_load_sentiment_sorted(self, synthetic_sentiment):
        df = sl.load_sentiment_csv(synthetic_sentiment)
        assert df["date"].is_monotonic_increasing

    def test_load_finance_columns(self, synthetic_finance):
        df = sl.load_finance_csv(synthetic_finance)
        assert "Date" in df.columns
        assert "Close" in df.columns

    def test_load_finance_numeric(self, synthetic_finance):
        df = sl.load_finance_csv(synthetic_finance)
        assert pd.api.types.is_float_dtype(df["Close"])

    def test_merge_returns_computed(self, merged_df):
        assert "ret_1d" in merged_df.columns
        assert "ret_5d" in merged_df.columns
        assert "ret_40d" in merged_df.columns

    def test_merge_row_count(self, merged_df):
        assert len(merged_df) > 0

    def test_load_sentiment_missing_col(self, tmp_path):
        df = pd.DataFrame({"date": ["2024-01-01"], "score": [0.5]})
        p = tmp_path / "bad.csv"
        df.to_csv(p, index=False)
        with pytest.raises((ValueError, KeyError)):
            sl.load_sentiment_csv(str(p))


# ─────────────────────────────────────────────────────────────────────────────
# 4.1 ADF
# ─────────────────────────────────────────────────────────────────────────────

class TestADF:
    def test_returns_dataframe(self, merged_df):
        result = sl.test_adf_stationarity(merged_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, merged_df):
        result = sl.test_adf_stationarity(merged_df)
        assert {"Series", "ADF_stat", "p_value", "is_stationary"}.issubset(result.columns)

    def test_two_rows(self, merged_df):
        result = sl.test_adf_stationarity(merged_df)
        assert len(result) == 2

    def test_p_value_in_range(self, merged_df):
        result = sl.test_adf_stationarity(merged_df)
        assert (result["p_value"] >= 0).all() and (result["p_value"] <= 1).all()


# ─────────────────────────────────────────────────────────────────────────────
# 4.2 Pearson / Spearman
# ─────────────────────────────────────────────────────────────────────────────

class TestPearsonSpearman:
    def test_returns_dataframe(self, merged_df):
        result = sl.test_pearson_spearman(merged_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, merged_df):
        result = sl.test_pearson_spearman(merged_df)
        for col in ["Orizzonte", "Campione", "N", "Pearson_r", "Pearson_p",
                    "Spearman_r", "Spearman_p"]:
            assert col in result.columns

    def test_pearson_r_bounded(self, merged_df):
        result = sl.test_pearson_spearman(merged_df)
        assert (result["Pearson_r"].abs() <= 1.0).all()

    def test_two_campioni(self, merged_df):
        result = sl.test_pearson_spearman(merged_df)
        assert set(result["Campione"]) == {"Tutti", "Sent!=0"}


# ─────────────────────────────────────────────────────────────────────────────
# 4.3 OLS
# ─────────────────────────────────────────────────────────────────────────────

class TestOLS:
    def test_returns_dataframe(self, merged_df):
        result = sl.test_ols_regression(merged_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_r2(self, merged_df):
        result = sl.test_ols_regression(merged_df)
        assert "R2" in result.columns
        assert (result["R2"] >= 0).all()

    def test_beta_is_float(self, merged_df):
        result = sl.test_ols_regression(merged_df)
        assert pd.api.types.is_float_dtype(result["Beta"])


# ─────────────────────────────────────────────────────────────────────────────
# 4.4 Lead-Lag
# ─────────────────────────────────────────────────────────────────────────────

class TestLeadLag:
    def test_returns_dataframe(self, merged_df):
        result = sl.test_lead_lag(merged_df)
        assert isinstance(result, pd.DataFrame)

    def test_lag_range(self, merged_df):
        result = sl.test_lead_lag(merged_df, lag_range=(-5, 5))
        assert result["Lag"].min() >= -5
        assert result["Lag"].max() <= 5

    def test_has_lag_zero(self, merged_df):
        result = sl.test_lead_lag(merged_df)
        assert 0 in result["Lag"].values


# ─────────────────────────────────────────────────────────────────────────────
# 4.5 Granger
# ─────────────────────────────────────────────────────────────────────────────

class TestGranger:
    def test_returns_dataframe(self, merged_df):
        result = sl.test_granger_causality(merged_df)
        assert isinstance(result, pd.DataFrame)

    def test_two_directions(self, merged_df):
        result = sl.test_granger_causality(merged_df)
        assert len(result["Direzione"].unique()) == 2

    def test_lags_present(self, merged_df):
        result = sl.test_granger_causality(merged_df, maxlag=3)
        assert set(result["Lag"]) == {1, 2, 3}

    def test_p_value_in_range(self, merged_df):
        result = sl.test_granger_causality(merged_df)
        assert (result["p_value"] >= 0).all() and (result["p_value"] <= 1).all()


# ─────────────────────────────────────────────────────────────────────────────
# 4.6 Event-Based
# ─────────────────────────────────────────────────────────────────────────────

class TestEventBased:
    def test_returns_dataframe(self, merged_df):
        result = sl.test_event_based(merged_df)
        assert isinstance(result, pd.DataFrame)

    def test_three_regimes(self, merged_df):
        result = sl.test_event_based(merged_df)
        assert set(result["Regime"]) == {"HighBull", "HighBear", "Neutro"}

    def test_ret_medio_is_percent(self, merged_df):
        result = sl.test_event_based(merged_df)
        # Returns expressed as % — should generally be between -100 and +100
        assert (result["Ret_medio_%"].abs() < 200).all()


# ─────────────────────────────────────────────────────────────────────────────
# 4.7 T-Test Bull vs Bear
# ─────────────────────────────────────────────────────────────────────────────

class TestTTestBullBear:
    def test_returns_dataframe(self, merged_df):
        result = sl.test_ttest_bull_vs_bear(merged_df)
        assert isinstance(result, pd.DataFrame)

    def test_diff_equals_bull_minus_bear(self, merged_df):
        result = sl.test_ttest_bull_vs_bear(merged_df)
        # Diff_% = Bull_mean_% - Bear_mean_% (rounded independently, allow ±0.01 tolerance)
        computed = result["Bull_mean_%"] - result["Bear_mean_%"]
        assert (result["Diff_%"] - computed).abs().max() < 0.02

    def test_p_value_in_range(self, merged_df):
        result = sl.test_ttest_bull_vs_bear(merged_df)
        assert (result["p_value"] >= 0).all() and (result["p_value"] <= 1).all()


# ─────────────────────────────────────────────────────────────────────────────
# 4.8 Backtest
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktest:
    def test_returns_dataframe(self, merged_df):
        result = sl.test_backtest_strategies(merged_df)
        assert isinstance(result, pd.DataFrame)

    def test_three_strategies(self, merged_df):
        result = sl.test_backtest_strategies(merged_df)
        assert set(result["Strategia"]) == {"Long Sent>Q70", "Long/Short", "Buy & Hold"}

    def test_has_sharpe(self, merged_df):
        result = sl.test_backtest_strategies(merged_df)
        assert "Sharpe" in result.columns

    def test_maxdd_negative(self, merged_df):
        result = sl.test_backtest_strategies(merged_df)
        assert (result["MaxDD_%"] <= 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# 4.9 Rolling Correlation
# ─────────────────────────────────────────────────────────────────────────────

class TestRollingCorrelation:
    def test_returns_tuple(self, merged_df):
        result = sl.test_rolling_correlation(merged_df)
        assert isinstance(result, tuple) and len(result) == 2

    def test_rolling_df_columns(self, merged_df):
        rolling_df, _ = sl.test_rolling_correlation(merged_df)
        for col in ["date", "rolling_r", "p_value", "n_nz"]:
            assert col in rolling_df.columns

    def test_rolling_r_bounded(self, merged_df):
        rolling_df, _ = sl.test_rolling_correlation(merged_df)
        assert (rolling_df["rolling_r"].abs() <= 1.0).all()

    def test_summary_has_five_rows(self, merged_df):
        _, summary = sl.test_rolling_correlation(merged_df)
        assert len(summary) == 5
