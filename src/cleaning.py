from __future__ import annotations
import numpy as np
import pandas as pd
import holidays


REQUIRED_COLUMNS = [
    "Planned Delivery Date",
    "Arrival Date",
    "Ordered Quantity",
    "Delivered Quantity",
    "Supplier",
]


def _to_datetime_safe(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def clean_data(
    df: pd.DataFrame,
    min_year: int = 2017,
    max_year: int = 2027,
    holiday_country: str = "DE",
) -> pd.DataFrame:
    """
    Data cleaning + anomaly flags.
    Mirrors your notebook logic, but makes it reusable and open-source friendly.
    """
    df = df.copy()

    # --- Parse dates ---
    for col in ["Planned Delivery Date", "Arrival Date"]:
        if col in df.columns:
            df[col] = _to_datetime_safe(df[col])

    # --- Core derived metrics ---
    df["days_diff"] = (df["Arrival Date"] - df["Planned Delivery Date"]).dt.days
    df["qty_diff"] = df["Delivered Quantity"] - df["Ordered Quantity"]

    # qty_deviation = (Delivered - Ordered)/Ordered ; protect division by 0
    denom = df["Ordered Quantity"].replace(0, np.nan)
    df["qty_deviation"] = (df["Delivered Quantity"] - df["Ordered Quantity"]) / denom

    # --- Anomaly flags ---
    # 1) Missing arrival date -> open/undelivered order
    df["is_open_order"] = df["Arrival Date"].isna().astype(int)

    # 2) Ordered Quantity == 0 or NaN -> invalid
    df["is_invalid"] = (df["Ordered Quantity"].isna() | (df["Ordered Quantity"] <= 0)).astype(int)

    # 3) Weekend & holiday flags (based on planned date)
    years = range(min_year, max_year + 1)
    cal = holidays.CountryHoliday(holiday_country, years=years)

    df["is_weekend"] = df["Planned Delivery Date"].dt.weekday >= 5
    df["is_holiday"] = df["Planned Delivery Date"].isin(cal)

    # 4) Extreme date anomalies
    planned_year = df["Planned Delivery Date"].dt.year
    mask_extreme = (planned_year < min_year) | (planned_year > max_year)
    df["is_extreme_date"] = mask_extreme.astype(int)

    # Optional Quality column handling
    if "Quality" in df.columns:
        df["Quality"] = df["Quality"].fillna("Unknown")

    return df
