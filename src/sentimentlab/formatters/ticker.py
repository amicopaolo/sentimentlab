"""Ticker symbol normalization across exchanges and data providers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Known exchange suffix mappings (Yahoo Finance → standard MIC)
_EXCHANGE_SUFFIX_MAP: dict[str, str] = {
    ".MI": "Borsa Italiana",     # Milan
    ".PA": "Euronext Paris",
    ".AS": "Euronext Amsterdam",
    ".L":  "London Stock Exchange",
    ".DE": "Xetra / Frankfurt",
    ".SW": "SIX Swiss Exchange",
    ".TO": "Toronto Stock Exchange",
    ".AX": "ASX Australia",
    ".HK": "Hong Kong Stock Exchange",
    ".T":  "Tokyo Stock Exchange",
    ".SS": "Shanghai Stock Exchange",
    ".SZ": "Shenzhen Stock Exchange",
    ".BO": "Bombay Stock Exchange",
    ".NS": "National Stock Exchange India",
    ".SA": "B3 Brazil",
}

# Crypto base currency pairs
_CRYPTO_QUOTE_CURRENCIES = {"USDT", "USD", "BTC", "ETH", "BNB", "USDC"}


@dataclass
class TickerFormatter:
    """
    Normalizes ticker symbols from various sources into canonical form.

    Parameters
    ----------
    uppercase : bool
        Convert ticker to uppercase (default True).
    strip_exchange : bool
        Remove exchange suffix like '.MI', '.L' (default False).
    provider : str
        Source provider hint: 'yahoo', 'bloomberg', 'binance', 'generic'.

    Example
    -------
    >>> fmt = TickerFormatter(provider="yahoo")
    >>> fmt.normalize("eni.mi")
    'ENI.MI'
    >>> TickerFormatter(strip_exchange=True).normalize("eni.MI")
    'ENI'
    """

    uppercase: bool = True
    strip_exchange: bool = False
    provider: str = "generic"

    def normalize(self, ticker: str) -> str:
        """Normalize a single ticker string."""
        return normalize_ticker(
            ticker,
            uppercase=self.uppercase,
            strip_exchange=self.strip_exchange,
            provider=self.provider,
        )

    def normalize_list(self, tickers: list[str]) -> list[str]:
        """Normalize a list of ticker strings."""
        return [self.normalize(t) for t in tickers]

    def get_exchange(self, ticker: str) -> str | None:
        """Extract the exchange name from a Yahoo-style suffix."""
        ticker = ticker.upper()
        for suffix, exchange in _EXCHANGE_SUFFIX_MAP.items():
            if ticker.endswith(suffix.upper()):
                return exchange
        return None

    def is_crypto(self, ticker: str) -> bool:
        """Heuristic check if ticker looks like a crypto pair."""
        ticker = ticker.upper().replace("-", "").replace("/", "")
        for quote in _CRYPTO_QUOTE_CURRENCIES:
            if ticker.endswith(quote) and len(ticker) > len(quote):
                return True
        return False


def normalize_ticker(
    ticker: str,
    uppercase: bool = True,
    strip_exchange: bool = False,
    provider: str = "generic",
) -> str:
    """
    Normalize a ticker symbol string.

    Parameters
    ----------
    ticker : str
        Raw ticker (e.g., 'aapl', 'ENI.MI', 'BTC-USD').
    uppercase : bool
        Force uppercase (default True).
    strip_exchange : bool
        Remove exchange suffix (default False).
    provider : str
        Hint for provider-specific normalization.

    Returns
    -------
    str

    Example
    -------
    >>> normalize_ticker("eni.mi", strip_exchange=True)
    'ENI'
    >>> normalize_ticker("btc-usd", provider="yahoo")
    'BTC-USD'
    """
    ticker = ticker.strip()

    if uppercase:
        ticker = ticker.upper()

    if provider == "binance":
        # Binance uses no separator: BTCUSDT → BTC-USDT
        ticker = ticker.replace("/", "").replace("-", "")
    elif provider == "yahoo":
        # Yahoo uses dash for crypto: BTC-USD
        pass

    if strip_exchange:
        for suffix in _EXCHANGE_SUFFIX_MAP:
            if ticker.upper().endswith(suffix.upper()):
                ticker = ticker[: -len(suffix)]
                break

    # Remove any trailing/leading whitespace again after manipulation
    return ticker.strip()
