from __future__ import annotations
import numpy as np
import pandas as pd


def add_comment_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 'Comment' column to track anomalies for later visualization and analysis.
    """
    df = df.copy()
    df["Comment"] = ""

    def _append(mask: pd.Series, text: str) -> None:
        df.loc[mask, "Comment"] = df.loc[mask, "Comment"] + text

    _append(df["is_open_order"] == 1, "Disappearing Order; ")
    _append(df["is_invalid"] == 1, "Invalid Order; ")
    _append(df["is_weekend"] == 1, "Weekend; ")
    _append(df["is_holiday"] == 1, "Holiday; ")
    _append(df["is_extreme_date"] == 1, "Extreme Date Anomalies; ")

    return df


def assign_delivery_status(days_diff: float, threshold: int = 1) -> str | float:
    """
    Delivery status label:
    - Early: days_diff < -threshold
    - On-time: -threshold <= days_diff <= threshold
    - Late: days_diff > threshold
    """
    if pd.isna(days_diff):
        return np.nan
    if days_diff < -threshold:
        return "Early"
    if -threshold <= days_diff <= threshold:
        return "On-time"
    return "Late"


def add_labels(df: pd.DataFrame, delivery_threshold: int = 1) -> pd.DataFrame:
    """
    Add:
    - delivery_status (Early/On-time/Late)
    - quantity_status (Less/Correct/More)
    """
    df = df.copy()
    df["delivery_status"] = df["days_diff"].apply(lambda x: assign_delivery_status(x, threshold=delivery_threshold))
    df["quantity_status"] = df["qty_diff"].apply(
        lambda x: "Less" if x < 0 else ("More" if x > 0 else "Correct")
    )
    return df
