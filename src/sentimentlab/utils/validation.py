"""OHLCV DataFrame validation with structured error reporting."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ValidationResult:
    """Result object returned by :func:`validate_ohlcv`."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        parts = [f"ValidationResult({status})"]
        for e in self.errors:
            parts.append(f"  ERROR   : {e}")
        for w in self.warnings:
            parts.append(f"  WARNING : {w}")
        return "\n".join(parts)

    def raise_if_invalid(self) -> None:
        """Raise ValueError if validation failed."""
        if not self.is_valid:
            raise ValueError(f"OHLCV validation failed:\n" + "\n".join(self.errors))


def validate_ohlcv(
    df: pd.DataFrame,
    require_volume: bool = True,
    max_nan_pct: float = 0.05,
    check_ohlc_logic: bool = True,
) -> ValidationResult:
    """
    Validate an OHLCV DataFrame for common data quality issues.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate (should have DatetimeIndex after format_ohlcv).
    require_volume : bool
        Fail if volume column is missing.
    max_nan_pct : float
        Maximum acceptable fraction of NaN values per price column (default 5%).
    check_ohlc_logic : bool
        Check that high >= open/close and low <= open/close.

    Returns
    -------
    ValidationResult

    Example
    -------
    >>> result = dfm.validate_ohlcv(df)
    >>> result.raise_if_invalid()
    >>> print(result)
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Check DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("Index is not a DatetimeIndex. Run format_ohlcv() first.")

    # 2. Check required price columns
    required_price = ["open", "high", "low", "close"]
    missing_cols = [c for c in required_price if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # 3. Check volume
    if require_volume and "volume" not in df.columns:
        errors.append("Column 'volume' is required but missing.")

    # 4. NaN check per column
    for col in required_price:
        if col in df.columns:
            nan_pct = df[col].isna().mean()
            if nan_pct > max_nan_pct:
                warnings.append(
                    f"Column '{col}' has {nan_pct:.1%} NaN values (threshold: {max_nan_pct:.0%})."
                )

    # 5. OHLC logic check
    if check_ohlc_logic and not missing_cols:
        bad_high = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
        bad_low = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
        if bad_high > 0:
            warnings.append(f"{bad_high} rows where high < max(open, close).")
        if bad_low > 0:
            warnings.append(f"{bad_low} rows where low > min(open, close).")

    # 6. Negative prices
    for col in required_price:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                errors.append(f"Column '{col}' has {neg_count} negative values.")

    # 7. Duplicate index
    if isinstance(df.index, pd.DatetimeIndex):
        dups = df.index.duplicated().sum()
        if dups > 0:
            warnings.append(f"{dups} duplicate timestamps in index.")

    # 8. Empty DataFrame
    if len(df) == 0:
        errors.append("DataFrame is empty.")

    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
