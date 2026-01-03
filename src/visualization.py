from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compare_missing_zero_counts(df_a: pd.DataFrame, df_b: pd.DataFrame, title_suffix: str = "") -> None:
    all_cols = sorted(list(set(df_a.columns).union(set(df_b.columns))))

    nan_counts_a = df_a.reindex(columns=all_cols).isna().sum()
    nan_counts_b = df_b.reindex(columns=all_cols).isna().sum()
    zero_counts_a = df_a.reindex(columns=all_cols).eq(0).sum()
    zero_counts_b = df_b.reindex(columns=all_cols).eq(0).sum()

    x = np.arange(len(all_cols))

    plt.figure(figsize=(min(18, 1.2 * len(all_cols)), 5))
    plt.bar(x - 0.18, nan_counts_a, width=0.35, label="Company A")
    plt.bar(x + 0.18, nan_counts_b, width=0.35, label="Company B")
    plt.xticks(x, all_cols, rotation=45, ha="right")
    plt.ylabel("Missing Values")
    plt.title(f"Missing Values by Company{title_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(min(18, 1.2 * len(all_cols)), 5))
    plt.bar(x - 0.18, zero_counts_a, width=0.35, label="Company A")
    plt.bar(x + 0.18, zero_counts_b, width=0.35, label="Company B")
    plt.xticks(x, all_cols, rotation=45, ha="right")
    plt.ylabel("Zero Values")
    plt.title(f"Zero Values by Company{title_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_special_orders(df: pd.DataFrame, company: str = "A") -> None:
    undelivered = df["Delivered Quantity"].isna()
    invalid = df["Ordered Quantity"].isna() | (df["Ordered Quantity"] == 0)

    counts = {
        "Undelivered": int(undelivered.sum()),
        "Invalid": int(invalid.sum()),
        "Total": int(len(df)),
    }

    plt.figure(figsize=(5, 5))
    plt.pie(
        [counts["Undelivered"], counts["Invalid"], counts["Total"] - counts["Undelivered"] - counts["Invalid"]],
        labels=["Undelivered", "Invalid", "Valid"],
        autopct="%1.1f%%",
    )
    plt.title(f"[{company}] Special Orders Distribution")
    plt.show()

    print(
        f"[{company}] Undelivered: {counts['Undelivered']} | Invalid: {counts['Invalid']} | Total: {counts['Total']}"
    )


def delivery_delay_comparison(df: pd.DataFrame, company: str = "A", filter_range: tuple[int, int] = (-80, 80)) -> None:
    df = df.copy()

    if "days_diff" not in df.columns:
        df["days_diff"] = (pd.to_datetime(df["Arrival Date"]) - pd.to_datetime(df["Planned Delivery Date"])).dt.days

    planned_year = pd.to_datetime(df["Planned Delivery Date"], errors="coerce").dt.year
    mask_extreme = (planned_year < 2017) | (planned_year > 2027) | (planned_year.isna())
    extreme_ratio = float(mask_extreme.mean())
    print(f"[{company}] Abnormal date ratio (<2017 or >2027 or missing): {extreme_ratio:.2%}")

    df_filtered = df.loc[~mask_extreme].copy()
    df_filtered = df_filtered[df_filtered["days_diff"].between(filter_range[0], filter_range[1])]
    print(
        f"[{company}] After filtering abnormal dates and outlier delays: "
        f"{len(df_filtered)} records remain ({len(df_filtered)/len(df):.2%} of total)"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(df["days_diff"].dropna(), bins=40, kde=True, ax=axes[0])
    axes[0].set_title(f"[{company}] All Data")
    axes[0].set_xlabel("days_diff (Arrival - Planned)")

    sns.histplot(df_filtered["days_diff"].dropna(), bins=40, kde=True, ax=axes[1])
    axes[1].set_title(f"[{company}] Filtered ({filter_range[0]} to {filter_range[1]} days)")
    axes[1].set_xlabel("days_diff (Arrival - Planned)")

    plt.tight_layout()
    plt.show()


def supplier_performance_analysis(df: pd.DataFrame, company: str = "A", top_n: int = 20, threshold: int = 1) -> pd.DataFrame:
    """
    Analyze supplier performance.
    - Early delivery is NOT acceptable (tracked separately)
    Returns supplier_stats table for further use.
    """
    df = df.copy()
    valid = df[(~df["Delivered Quantity"].isna()) & (~df["Ordered Quantity"].isna()) & (df["Ordered Quantity"] > 0)].copy()

    valid["days_diff"] = (pd.to_datetime(valid["Arrival Date"]) - pd.to_datetime(valid["Planned Delivery Date"])).dt.days
    valid["status"] = valid["days_diff"].apply(lambda x: "Late" if x > threshold else ("Early" if x < -threshold else "On-time"))

    supplier_stats = valid.groupby("Supplier")["status"].value_counts(normalize=True).unstack().fillna(0)
    supplier_stats["Total Orders"] = valid.groupby("Supplier").size()

    supplier_stats_top = supplier_stats.sort_values("Total Orders", ascending=False).head(top_n)

    supplier_stats_top[["Late", "On-time", "Early"]].plot(kind="bar", stacked=True, figsize=(12, 5))
    plt.title(f"[{company}] Top {top_n} Suppliers Delivery Performance")
    plt.ylabel("Proportion")
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.show()

    # Simple screening rules (can be customized)
    if company.upper() == "A":
        high_risk = supplier_stats[(supplier_stats.get("On-time", 0) == 0.0) & (supplier_stats["Total Orders"] >= 10)]
        best = supplier_stats[(supplier_stats.get("On-time", 0) == 1.0) & (supplier_stats["Total Orders"] >= 10)]
        print(f"\n[{company}] High-risk suppliers (on-time = 0%, orders >= 10): {len(high_risk)}")
        print(f"[{company}] Best suppliers (on-time = 100%, orders >= 10): {len(best)}")
    else:
        high_risk = supplier_stats[(supplier_stats.get("Late", 0) == 1.0) & (supplier_stats["Total Orders"] >= 10)]
        best = supplier_stats[(supplier_stats.get("Late", 0) == 0.0) & (supplier_stats["Total Orders"] >= 10)]
        print(f"\n[{company}] High-risk suppliers (late = 100%, orders >= 10): {len(high_risk)}")
        print(f"[{company}] Best suppliers (late = 0%, orders >= 10): {len(best)}")

    return supplier_stats
