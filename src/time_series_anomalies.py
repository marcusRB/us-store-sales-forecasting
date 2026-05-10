import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from statsmodels.tsa.seasonal import seasonal_decompose

np.random.seed(42)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots", "time_series")
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)


def load_daily_event_frame() -> pd.DataFrame:
    sales = pl.read_parquet(os.path.join(DATA_DIR, "sales.parquet")).to_pandas()
    sales["date"] = pd.to_datetime(sales["date"])
    sales["zero_sale_flag"] = sales["zero_sale_reason"].notna().astype(int)
    sales["promotion_flag"] = sales["is_promo_day"].fillna(False).astype(int)
    sales["holiday_flag"] = sales["is_holiday"].fillna(False).astype(int)
    sales["macro_flag"] = sales["macro_event_id"].notna().astype(int)
    sales["micro_flag"] = sales["micro_event_id"].notna().astype(int)
    sales["peak_flag"] = sales["is_peak_day"].fillna(False).astype(int)
    sales["stockout_flag"] = sales["is_stockout"].fillna(False).astype(int)

    daily = (
        sales.groupby("date", as_index=False)
        .agg(
            total_units_sold=("units_sold", "sum"),
            rows=("store_id", "count"),
            zero_sale_events=("zero_sale_flag", "sum"),
            stockout_events=("stockout_flag", "sum"),
            holiday_rows=("holiday_flag", "sum"),
            promo_rows=("promotion_flag", "sum"),
            macro_rows=("macro_flag", "sum"),
            micro_rows=("micro_flag", "sum"),
            peak_rows=("peak_flag", "sum"),
            avg_true_demand=("true_demand", "mean"),
            avg_inflation_rate=("inflation_rate", "mean"),
            avg_precipitation_mm=("precipitation_mm", "mean"),
        )
        .sort_values("date")
        .set_index("date")
        .asfreq("D")
        .fillna(0)
    )

    daily["rolling_mean_14d"] = daily["total_units_sold"].rolling(window=14, min_periods=7).mean()
    daily["rolling_std_14d"] = daily["total_units_sold"].rolling(window=14, min_periods=7).std().replace(0, np.nan)
    daily["zscore_14d"] = (daily["total_units_sold"] - daily["rolling_mean_14d"]) / daily["rolling_std_14d"]
    daily["zero_sale_ratio"] = daily["zero_sale_events"] / daily["rows"].replace(0, np.nan)
    daily["peak_ratio"] = daily["peak_rows"] / daily["rows"].replace(0, np.nan)
    daily = daily.fillna(0)
    return daily


def classify_daily_anomaly(row: pd.Series) -> str:
    if row["holiday_rows"] > 0 and row["zscore_14d"] >= 1.75:
        return "holiday_peak"
    if row["promo_rows"] > 0 and row["zscore_14d"] >= 1.75:
        return "promotion_peak"
    if row["macro_rows"] > 0 and row["zscore_14d"] <= -1.5:
        return "macro_stress_drop"
    if row["micro_rows"] > 0 and row["zero_sale_ratio"] >= 0.08:
        return "micro_disruption"
    if row["zero_sale_ratio"] >= 0.12 or row["stockout_events"] >= 15:
        return "zero_sale_cluster"
    if row["peak_ratio"] >= 0.08 and row["zscore_14d"] >= 2.0:
        return "peak_surge"
    if abs(row["zscore_14d"]) >= 2.5:
        return "statistical_outlier"
    return "normal"


def plot_decomposition(daily_sales: pd.DataFrame) -> None:
    period = 7 if len(daily_sales) >= 14 else max(2, len(daily_sales) // 2)
    if period <= 1:
        return
    decomposition = seasonal_decompose(daily_sales["total_units_sold"], model="additive", period=period)
    decomposition.plot()
    plt.savefig(os.path.join(PLOTS_DIR, "ts_decomposition.png"), bbox_inches="tight")
    plt.close()


def plot_anomalies(daily_sales: pd.DataFrame, anomalies: pd.DataFrame) -> None:
    colors = {
        "holiday_peak": "tab:green",
        "promotion_peak": "tab:orange",
        "macro_stress_drop": "tab:red",
        "micro_disruption": "tab:brown",
        "zero_sale_cluster": "tab:purple",
        "peak_surge": "tab:blue",
        "statistical_outlier": "tab:gray",
    }
    plt.figure(figsize=(16, 7))
    plt.plot(daily_sales.index, daily_sales["total_units_sold"], label="Daily units sold", linewidth=1.5)
    for cause, cause_df in anomalies.groupby("anomaly_cause"):
        plt.scatter(
            cause_df.index,
            cause_df["total_units_sold"],
            label=cause,
            s=26,
            alpha=0.8,
            color=colors.get(cause, "black"),
        )
    plt.title("Daily Sales Anomalies With Event Attribution")
    plt.xlabel("Date")
    plt.ylabel("Units sold")
    plt.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "anomalies_by_cause.png"))
    plt.close()


def save_outputs(daily_sales: pd.DataFrame, anomalies: pd.DataFrame) -> None:
    daily_sales.to_csv(os.path.join(ANALYSIS_DIR, "daily_event_summary.csv"))
    anomalies.to_csv(os.path.join(ANALYSIS_DIR, "daily_anomalies.csv"))

    summary = anomalies.groupby("anomaly_cause").agg(
        count=("total_units_sold", "count"),
        avg_units_sold=("total_units_sold", "mean"),
        avg_zero_sale_ratio=("zero_sale_ratio", "mean"),
    )
    summary.to_csv(os.path.join(ANALYSIS_DIR, "anomaly_cause_summary.csv"))

    with open(os.path.join(ANALYSIS_DIR, "anomaly_report.md"), "w", encoding="utf-8") as handle:
        handle.write("# Time Series Anomaly Analysis\n\n")
        handle.write("This report links daily sales anomalies to explicit simulated retail causes: holidays, promotions, macro stress, micro disruptions, peak surges, and clustered zero-sale events.\n\n")
        handle.write("## Detection Logic\n")
        handle.write("- Baseline anomaly score: 14-day rolling z-score on aggregated units sold.\n")
        handle.write("- Event-aware overrides: holiday, promotion, macro, micro, stockout, and peak-day indicators from the generator.\n")
        handle.write("- Zero-sale clusters: dates with an unusually high share of rows marked by explicit zero-sale reasons.\n\n")
        handle.write("## Summary\n")
        handle.write(summary.to_csv())
        handle.write("\n")


def analyze_ts_and_anomalies():
    daily_sales = load_daily_event_frame()
    plot_decomposition(daily_sales)
    daily_sales["anomaly_cause"] = daily_sales.apply(classify_daily_anomaly, axis=1)
    anomalies = daily_sales[daily_sales["anomaly_cause"] != "normal"].copy()
    plot_anomalies(daily_sales, anomalies)
    save_outputs(daily_sales, anomalies)
    print(f"TS and anomaly analysis complete. Found {len(anomalies)} event-aware anomalies.")
    print(f"Outputs saved in {ANALYSIS_DIR} and {PLOTS_DIR}")


if __name__ == "__main__":
    analyze_ts_and_anomalies()
