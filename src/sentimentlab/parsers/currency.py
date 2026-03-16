"""Currency detection and code normalization utilities."""

from __future__ import annotations

import re

# ISO 4217 currency codes (subset — major + common crypto)
_ISO4217: dict[str, str] = {
    "USD": "US Dollar", "EUR": "Euro", "GBP": "British Pound",
    "JPY": "Japanese Yen", "CHF": "Swiss Franc", "CAD": "Canadian Dollar",
    "AUD": "Australian Dollar", "NZD": "New Zealand Dollar",
    "CNY": "Chinese Yuan", "HKD": "Hong Kong Dollar", "SGD": "Singapore Dollar",
    "SEK": "Swedish Krona", "NOK": "Norwegian Krone", "DKK": "Danish Krone",
    "MXN": "Mexican Peso", "BRL": "Brazilian Real", "INR": "Indian Rupee",
    "RUB": "Russian Ruble", "ZAR": "South African Rand", "TRY": "Turkish Lira",
    "KRW": "South Korean Won", "PLN": "Polish Zloty", "CZK": "Czech Koruna",
    "HUF": "Hungarian Forint", "RON": "Romanian Leu",
    # Crypto
    "BTC": "Bitcoin", "ETH": "Ethereum", "BNB": "Binance Coin",
    "USDT": "Tether", "USDC": "USD Coin", "SOL": "Solana",
    "ADA": "Cardano", "XRP": "Ripple", "DOT": "Polkadot", "MATIC": "Polygon",
}

# Symbol → ISO code
_SYMBOL_MAP: dict[str, str] = {
    "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₣": "CHF",
    "₹": "INR", "₩": "KRW", "₺": "TRY", "R$": "BRL", "kr": "SEK",
}


def detect_currency(text: str) -> str | None:
    """
    Attempt to detect a currency code from a string.

    Checks for ISO 4217 codes and common currency symbols.

    Parameters
    ----------
    text : str
        Input string (e.g., column name, label, price string like "$42.00").

    Returns
    -------
    str | None
        ISO 4217 currency code (e.g., 'USD') or None if not detected.

    Example
    -------
    >>> detect_currency("Price in EUR")
    'EUR'
    >>> detect_currency("$42.50")
    'USD'
    """
    # Check symbols first
    for symbol, code in sorted(_SYMBOL_MAP.items(), key=lambda x: -len(x[0])):
        if symbol in text:
            return code

    # Check ISO codes (case-insensitive, word boundary)
    text_upper = text.upper()
    for code in _ISO4217:
        pattern = r'\b' + re.escape(code) + r'\b'
        if re.search(pattern, text_upper):
            return code

    return None


def convert_currency_code(code: str, to_format: str = "iso") -> str:
    """
    Convert a currency code between formats.

    Parameters
    ----------
    code : str
        Input currency code or symbol.
    to_format : str
        Target format: 'iso' (3-letter code), 'name' (full name), 'symbol'.

    Returns
    -------
    str
        Converted currency identifier.

    Raises
    ------
    ValueError
        If the code is not recognized.

    Example
    -------
    >>> convert_currency_code("€", to_format="iso")
    'EUR'
    >>> convert_currency_code("USD", to_format="name")
    'US Dollar'
    """
    # Normalize input
    code = code.strip()

    # Check if input is a symbol
    if code in _SYMBOL_MAP:
        iso = _SYMBOL_MAP[code]
    elif code.upper() in _ISO4217:
        iso = code.upper()
    else:
        raise ValueError(f"Unrecognized currency code or symbol: '{code}'")

    if to_format == "iso":
        return iso
    elif to_format == "name":
        return _ISO4217[iso]
    elif to_format == "symbol":
        reverse = {v: k for k, v in _SYMBOL_MAP.items()}
        return reverse.get(iso, iso)  # fallback to ISO if no symbol
    else:
        raise ValueError(f"Unknown to_format: '{to_format}'. Use 'iso', 'name', or 'symbol'.")


def list_supported_currencies() -> dict[str, str]:
    """Return all supported ISO 4217 codes and their full names."""
    return dict(_ISO4217)
