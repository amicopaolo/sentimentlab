# sentimentlab

A Python library for **sentiment-driven financial market analysis**.

Load daily sentiment scores (e.g. from FinBERT/GDELT), pair them with yfinance price data, and run a full statistical test suite — the same methodology used in the NVDA/GLD contrarian analysis research.

```bash
pip install sentimentlab
```

---

## Quick Start

```python
import sentimentlab as sl

# Load CSVs
sent   = sl.load_sentiment_csv("gdelt_daily_sentiment.csv")
prices = sl.load_finance_csv("yfinance_nvda_90d.csv")

# Merge and compute forward returns at horizons [1, 3, 5, 10, 15, 20, 30, 40] days
df = sl.merge_sentiment_finance(sent, prices)

# Run all 9 tests
print(sl.test_adf_stationarity(df))
print(sl.test_pearson_spearman(df))
print(sl.test_ols_regression(df))
print(sl.test_lead_lag(df))
print(sl.test_granger_causality(df))
print(sl.test_event_based(df))
print(sl.test_ttest_bull_vs_bear(df))
print(sl.test_backtest_strategies(df))
rolling_df, summary = sl.test_rolling_correlation(df)
print(summary)
```

---

## CLI

```bash
sentimentlab \
  --sentiment gdelt_daily_sentiment.csv \
  --finance yfinance_nvda_90d.csv \
  --horizons 1 5 10 20 \
  --test adf pearson ols granger backtest
```

Available `--test` values: `adf`, `pearson`, `ols`, `leadlag`, `granger`, `event`, `ttest`, `backtest`, `rolling`

---

## CSV Format

### Sentiment CSV
Expected columns (customizable via parameters):

| Day        | daily_sentiment |
|------------|----------------|
| 2024-01-15 | 0.7231         |
| 2024-01-16 | -0.4812        |
| 2024-01-17 | 0.0            |

```python
sl.load_sentiment_csv("file.csv", date_col="Day", sentiment_col="daily_sentiment")
```

### Finance CSV
Standard output of `yf.download(...).to_csv()`:

```python
import yfinance as yf
data = yf.download("NVDA", start="2024-01-01", end="2024-06-01", auto_adjust=False)
data.to_csv("prices.csv", index=True)
```

```python
sl.load_finance_csv("prices.csv", price_col="Close")
```

Both flat single-header and yfinance multi-header formats are auto-detected.

---

## Statistical Tests (sections 4.1–4.9)

| # | Function | What it does |
|---|----------|-------------|
| 4.1 | `test_adf_stationarity` | ADF unit-root test — checks both series are stationary |
| 4.2 | `test_pearson_spearman` | Pearson & Spearman r at each horizon, full + non-zero sentiment |
| 4.3 | `test_ols_regression` | OLS β, p-value, R² at each horizon |
| 4.4 | `test_lead_lag` | Cross-correlation at lags −15…+15 (who leads whom?) |
| 4.5 | `test_granger_causality` | Granger causality Sent→Ret and Ret→Sent |
| 4.6 | `test_event_based` | One-sample t-test: HighBull / HighBear / Neutro regimes × horizon |
| 4.7 | `test_ttest_bull_vs_bear` | Independent Welch t-test HighBull vs HighBear returns |
| 4.8 | `test_backtest_strategies` | Long / Long-Short / Buy&Hold: total return, Sharpe, max drawdown |
| 4.9 | `test_rolling_correlation` | Rolling Pearson r over sliding window (default 20 days) |

---

## OHLCV Formatting

```python
import sentimentlab as sl

# Normalize any OHLCV DataFrame (from yfinance, CSV, etc.)
clean = sl.format_ohlcv(raw_df, price_decimals=2, fill_missing=True)

# Validate data quality
result = sl.validate_ohlcv(clean)
result.raise_if_invalid()
print(result)

# Human-readable summary
print(sl.summary(clean, title="NVDA Daily"))
```

---

## Installation

### From PyPI (once published)
```bash
pip install sentimentlab
```

### With optional extras
```bash
pip install "sentimentlab[yfinance]"   # adds yfinance
pip install "sentimentlab[full]"        # adds yfinance + matplotlib + rich
```

### From source
```bash
git clone https://github.com/paolo-amicopk/sentimentlab
cd sentimentlab
pip install -e ".[dev]"
pytest
```

---

## Dependencies

| Package | Role |
|---------|------|
| `pandas` | DataFrames |
| `numpy` | Numerical ops |
| `scipy` | Pearson, Spearman, t-tests, OLS |
| `statsmodels` | ADF, Granger causality |
| `pytz` | Timezone handling |
| `python-dateutil` | Timestamp parsing |

Optional: `yfinance`, `matplotlib`, `rich`

---

## License

MIT © 2026 Paolo Amico
