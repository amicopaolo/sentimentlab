# sentimentlab

A Python library for **sentiment-driven financial market analysis**.

Load daily sentiment scores (e.g. from FinBERT/GDELT), pair them with yfinance price data,
and run a full statistical test suite — the same methodology used in the NVDA/GLD contrarian analysis.

```bash
pip install sentimentlab
```

---

## Tutorial (zero external files needed)

The fastest way to get started — no CSV files, no internet, no yfinance required:

```python
import sentimentlab as sl

# 1. Generate synthetic sample CSV files in the current directory
sent_path, prices_path = sl.make_sample_data()
# [sentimentlab] Sample data generated for 'NVDA'
#   Sentiment : .../sample_sentiment.csv  (90 days)
#   Prices    : .../sample_prices.csv     (135 days)

# 2. Load them
sent   = sl.load_sentiment_csv(sent_path)
prices = sl.load_finance_csv(prices_path)

# 3. Merge + compute forward returns at horizons [1,3,5,10,15,20,30,40] days
df = sl.merge_sentiment_finance(sent, prices)
print(df.shape)       # (90, 17)
print(df.columns.tolist())

# 4. Run the 9 statistical tests

# 4.1 — Are both series stationary? (required for valid regression)
print(sl.test_adf_stationarity(df))

# 4.2 — Pearson & Spearman correlations at each horizon
corr = sl.test_pearson_spearman(df)
print(corr[corr["Pearson_p"] < 0.05])   # significant horizons only

# 4.3 — OLS regression: β, p-value, R²
reg = sl.test_ols_regression(df)
print(reg[reg["p_beta"] < 0.05])

# 4.4 — Who leads whom? (lag −15 … +15)
lag = sl.test_lead_lag(df)
print(lag[lag["p_value"] < 0.05])

# 4.5 — Granger causality (bidirectional)
gc = sl.test_granger_causality(df)
print(gc[gc["p_value"] < 0.05])

# 4.6 — Event-based: HighBull / HighBear / Neutro regimes
ev = sl.test_event_based(df)
print(ev[(ev["Regime"] == "HighBear") & (ev["p_value"] < 0.05)])

# 4.7 — HighBull vs HighBear independent t-test
tt = sl.test_ttest_bull_vs_bear(df)
print(tt[tt["p_value"] < 0.05])

# 4.8 — Backtest: Long / Long-Short / Buy & Hold
bt = sl.test_backtest_strategies(df)
print(bt.to_string(index=False))

# 4.9 — Rolling Pearson r (window=20 days, vs ret_5d)
rolling_df, summary = sl.test_rolling_correlation(df)
print(summary.to_string(index=False))
```

### With your own data

Once you have real files, just swap the paths:

```python
import sentimentlab as sl

sent   = sl.load_sentiment_csv(
    "gdelt_events_90d_nvidia_daily_sentiment.csv",
    date_col="Day",
    sentiment_col="daily_sentiment",
)
prices = sl.load_finance_csv(
    "yfinance_nvda_90d.csv",
    price_col="Close",
)
df = sl.merge_sentiment_finance(sent, prices)
```

---

## CLI

Run all tests from the terminal without writing Python:

```bash
# With sample data (generates files on the fly)
python -c "import sentimentlab as sl; sl.make_sample_data()"
sentimentlab --sentiment sample_sentiment.csv --finance sample_prices.csv

# With real files, select specific tests
sentimentlab \
  --sentiment gdelt_daily_sentiment.csv \
  --finance   yfinance_nvda_90d.csv \
  --horizons  1 5 10 20 \
  --test      adf pearson ols granger backtest
```

Available `--test` values: `adf`, `pearson`, `ols`, `leadlag`, `granger`, `event`, `ttest`, `backtest`, `rolling`

---

## CSV Format

### Sentiment CSV

| Day        | daily_sentiment |
|------------|----------------|
| 2024-01-15 | 0.7231         |
| 2024-01-16 | -0.4812        |
| 2024-01-17 | 0.0            |

Column names are configurable:
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

Both flat single-header and yfinance multi-level-header formats are auto-detected.

---

## Statistical Tests Reference

| # | Function | Description |
|---|----------|-------------|
| 4.1 | `test_adf_stationarity` | ADF unit-root test — stationarity check |
| 4.2 | `test_pearson_spearman` | Pearson & Spearman r at multiple horizons |
| 4.3 | `test_ols_regression` | OLS β, p-value, R² at multiple horizons |
| 4.4 | `test_lead_lag` | Cross-correlation at lags −15…+15 |
| 4.5 | `test_granger_causality` | Granger causality in both directions |
| 4.6 | `test_event_based` | One-sample t-test: HighBull / HighBear / Neutro |
| 4.7 | `test_ttest_bull_vs_bear` | Welch t-test HighBull vs HighBear returns |
| 4.8 | `test_backtest_strategies` | Long / Long-Short / Buy&Hold metrics |
| 4.9 | `test_rolling_correlation` | Rolling Pearson r over sliding window |

---

## OHLCV Utilities

```python
import sentimentlab as sl

# Normalize any OHLCV DataFrame
clean = sl.format_ohlcv(raw_df, price_decimals=2, fill_missing=True)

# Validate data quality
result = sl.validate_ohlcv(clean)
result.raise_if_invalid()

# Human-readable summary
print(sl.summary(clean, title="NVDA Daily"))
```

---

## Installation

```bash
# Base install
pip install sentimentlab

# With yfinance support
pip install "sentimentlab[yfinance]"

# Full (+ matplotlib, rich)
pip install "sentimentlab[full]"

# Development
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
| `numpy` | Numerical operations |
| `scipy` | Pearson, Spearman, t-tests, OLS |
| `statsmodels` | ADF test, Granger causality |
| `pytz` | Timezone handling |
| `python-dateutil` | Timestamp parsing |

Optional: `yfinance`, `matplotlib`, `rich`

---

## License

MIT © 2026 Paolo Amico
