from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic features:
    - planned_month, planned_weekday
    - supplier_freq
    - article_prefix (first 3 chars)
    """
    df = df.copy()
    df["planned_month"] = df["Planned Delivery Date"].dt.month
    df["planned_weekday"] = df["Planned Delivery Date"].dt.weekday

    df["supplier_freq"] = df["Supplier"].map(df["Supplier"].value_counts())

    if "Product Article Number" in df.columns:
        df["article_prefix"] = df["Product Article Number"].astype(str).str[:3]
    else:
        df["article_prefix"] = "000"

    return df


def extended_feature_engineering_delivery(df: pd.DataFrame, encode_quality: bool = True) -> pd.DataFrame:
    """
    Additional aggregated features for delivery prediction:
    - supplier_delay_mean, supplier_late_rate, supplier_less_rate, supplier_more_rate
    - article_prefix_freq
    - planned_quarter, order counters
    - monthly supplier/product stats
    - optional quality_encoded (Company A only)
    """
    df = df.copy()

    # supplier aggregated
    df["supplier_delay_mean"] = df.groupby("Supplier")["days_diff"].transform("mean")
    df["supplier_late_rate"] = df.groupby("Supplier")["delivery_status"].transform(lambda x: (x == "Late").mean())
    df["supplier_less_rate"] = df.groupby("Supplier")["quantity_status"].transform(lambda x: (x == "Less").mean())
    df["supplier_more_rate"] = df.groupby("Supplier")["quantity_status"].transform(lambda x: (x == "More").mean())

    # article prefix freq
    df["article_prefix_freq"] = df.groupby("article_prefix")["article_prefix"].transform("count")

    # extra time features
    df["planned_quarter"] = df["planned_month"].apply(lambda x: (x - 1) // 3 + 1)

    # sequential counters (within group)
    df["supplier_order_count"] = df.groupby("Supplier").cumcount() + 1

    if "Product Article Number" in df.columns:
        df["product_order_count"] = df.groupby("Product Article Number").cumcount() + 1
        df["supplier_month_ontime_rate"] = df.groupby(["Supplier", "planned_month"])["delivery_status"].transform(
            lambda x: (x == "On-time").mean()
        )
        df["supplier_month_delay_mean"] = df.groupby(["Supplier", "planned_month"])["days_diff"].transform("mean")
        df["product_month_delay_mean"] = df.groupby(["Product Article Number", "planned_month"])["days_diff"].transform("mean")
    else:
        df["product_order_count"] = np.nan
        df["supplier_month_ontime_rate"] = np.nan
        df["supplier_month_delay_mean"] = np.nan
        df["product_month_delay_mean"] = np.nan

    # quality encoding if exists
    if encode_quality and "Quality" in df.columns:
        df["quality_encoded"] = LabelEncoder().fit_transform(df["Quality"].astype(str))
    else:
        df["quality_encoded"] = np.nan

    return df


def feature_engineering_quantity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features for quantity accuracy model:
    - supplier_quantity_deviation_mean
    - product_quantity_deviation_mean
    """
    df = df.copy()
    df["supplier_quantity_deviation_mean"] = df.groupby("Supplier")["qty_diff"].transform("mean")
    if "Product Article Number" in df.columns:
        df["product_quantity_deviation_mean"] = df.groupby("Product Article Number")["qty_diff"].transform("mean")
    else:
        df["product_quantity_deviation_mean"] = np.nan
    return df