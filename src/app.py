import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
TRAINING_RUNS_DIR = OUTPUT_DIR / "training_runs"


def safe_read_text(path: Path) -> str:
    if not path.exists():
        return "File not found."
    return path.read_text(encoding="utf-8")


def safe_read_csv(path: Path, rows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    dataframe = pd.read_csv(path)
    return dataframe.head(rows) if rows is not None else dataframe


def list_eda_runs() -> list[Path]:
    if not PLOTS_DIR.exists():
        return []
    runs = [child for child in PLOTS_DIR.iterdir() if child.is_dir() and child.name.replace("_", "").isdigit()]
    return sorted(runs, key=lambda item: item.name, reverse=True)


def build_log_index() -> pd.DataFrame:
    if not OUTPUT_DIR.exists():
        return pd.DataFrame(columns=["timestamp", "relative_path", "size_kb"])

    rows = []
    for path in OUTPUT_DIR.rglob("*"):
        if path.is_file():
            stat = path.stat()
            rows.append({
                "timestamp": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "relative_path": str(path.relative_to(ROOT_DIR)),
                "size_kb": round(stat.st_size / 1024, 2),
            })
    return pd.DataFrame(rows).sort_values("timestamp", ascending=False)


def list_training_runs() -> list[Path]:
    if not TRAINING_RUNS_DIR.exists():
        return []
    runs = [child for child in TRAINING_RUNS_DIR.iterdir() if child.is_dir() and child.name.replace("_", "").isdigit()]
    return sorted(runs, key=lambda item: item.name, reverse=True)


def load_run_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        import json
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def render_grouped_metric_table(run_dir: Path, title: str, filename: str, metric_default: str = "wape") -> None:
    file_path = run_dir / filename
    dataframe = safe_read_csv(file_path)
    st.subheader(title)
    if dataframe.empty:
        st.info(f"No data found for {filename}.")
        return

    metric_options = [column for column in ["wape", "rmse", "mae", "smape", "mape", "bias", "r2"] if column in dataframe.columns]
    selected_metric = st.selectbox(
        f"Sort metric for {title}",
        metric_options,
        index=metric_options.index(metric_default) if metric_default in metric_options else 0,
        key=f"{title}_{run_dir.name}",
    )
    ascending = selected_metric not in {"r2", "actual_sum", "predicted_sum", "count"}
    ranked = dataframe.sort_values(selected_metric, ascending=ascending)
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Best combinations")
        st.dataframe(ranked.head(25), width="stretch", hide_index=True)
    with col2:
        st.caption("Worst combinations")
        st.dataframe(ranked.tail(25).sort_values(selected_metric, ascending=not ascending), width="stretch", hide_index=True)


def render_overview() -> None:
    st.title("US Store Sales Forecasting Dashboard")
    st.markdown(
        "This dashboard reads the generated artifacts from the `output/` folder and lets you inspect EDA, time-series anomalies, interpolation tables, and forecasting results."
    )

    metrics_path = OUTPUT_DIR / "ml_interpolation_metrics.txt"
    stat_path = OUTPUT_DIR / "statistical_info.txt"
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Latest Statistical Summary")
        st.code(safe_read_text(stat_path), language="text")
    with col2:
        st.subheader("Latest Forecasting Metrics")
        st.code(safe_read_text(metrics_path), language="text")

    st.subheader("Latest Output Files")
    log_df = build_log_index().head(12)
    st.dataframe(log_df, width="stretch", hide_index=True)

    training_runs = list_training_runs()
    if training_runs:
        latest_run = training_runs[0]
        manifest = load_run_manifest(latest_run)
        st.subheader("Latest Trainer Run")
        st.caption(f"Run timestamp: {latest_run.name}")
        if manifest.get("metrics"):
            metrics_df = pd.DataFrame(list(manifest["metrics"].items()), columns=["metric", "value"])
            st.dataframe(metrics_df, width="stretch", hide_index=True)


def render_eda() -> None:
    st.header("EDA")
    eda_runs = list_eda_runs()
    if not eda_runs:
        st.warning("No timestamped EDA runs were found in the output folder.")
        return

    selected_run = st.selectbox("Select EDA run timestamp", options=eda_runs, format_func=lambda item: item.name)
    st.caption(f"Selected run folder: {selected_run.relative_to(ROOT_DIR)}")

    stats_text = safe_read_text(OUTPUT_DIR / "statistical_info.txt")
    st.subheader("Dataset Statistics")
    st.code(stats_text, language="text")

    image_cols = st.columns(2)
    total_sales_plot = selected_run / "total_sales_over_time.png"
    sales_by_type_plot = selected_run / "sales_by_store_type.png"
    with image_cols[0]:
        if total_sales_plot.exists():
            st.image(str(total_sales_plot), caption="Total units sold over time", width="stretch")
    with image_cols[1]:
        if sales_by_type_plot.exists():
            st.image(str(sales_by_type_plot), caption="Sales by store type", width="stretch")


def render_time_series() -> None:
    st.header("Time Series and Anomalies")
    col1, col2 = st.columns(2)
    decomposition_plot = PLOTS_DIR / "time_series" / "ts_decomposition.png"
    anomaly_plot = PLOTS_DIR / "time_series" / "anomalies_by_cause.png"
    with col1:
        if decomposition_plot.exists():
            st.image(str(decomposition_plot), caption="Seasonal decomposition", width="stretch")
    with col2:
        if anomaly_plot.exists():
            st.image(str(anomaly_plot), caption="Anomalies by cause", width="stretch")

    st.subheader("Anomaly Report")
    st.markdown(safe_read_text(ANALYSIS_DIR / "anomaly_report.md"))

    st.subheader("Anomaly Cause Summary")
    summary_df = safe_read_csv(ANALYSIS_DIR / "anomaly_cause_summary.csv")
    if not summary_df.empty:
        st.dataframe(summary_df, width="stretch", hide_index=True)

    st.subheader("Sample Daily Anomalies")
    anomalies_df = safe_read_csv(ANALYSIS_DIR / "daily_anomalies.csv", rows=200)
    if not anomalies_df.empty:
        st.dataframe(anomalies_df, width="stretch")


def render_interpolation() -> None:
    st.header("Interpolation")
    st.markdown(
        "The interpolation stage builds a denormalized modeling table and stores masked-row evaluation outputs for downstream imputation analysis."
    )

    training_runs = list_training_runs()
    selected_run = None
    if training_runs:
        selected_run = st.selectbox("Select trainer run", options=training_runs, format_func=lambda item: item.name, key="interpolation_run")
        st.caption(f"Selected trainer run folder: {selected_run.relative_to(ROOT_DIR)}")

    sample_df = safe_read_csv(OUTPUT_DIR / "interpolation_modeling_sample.csv", rows=200)
    if not sample_df.empty:
        st.subheader("Modeling Table Sample")
        st.dataframe(sample_df, width="stretch")

    masked_path = (selected_run / "interpolation_masked_rows.csv") if selected_run else (OUTPUT_DIR / "interpolation_masked_rows.csv")
    masked_df = safe_read_csv(masked_path, rows=200)
    if not masked_df.empty:
        st.subheader("Masked Interpolation Evaluation Sample")
        st.dataframe(masked_df, width="stretch")

    interpolation_plot = (selected_run / "interpolation_masked_test_rows.png") if selected_run and (selected_run / "interpolation_masked_test_rows.png").exists() else (PLOTS_DIR / "ml_forecasting" / "interpolation_masked_test_rows.png")
    legacy_plot = PLOTS_DIR / "ml_forecasting" / "interpolation_results.png"
    if interpolation_plot.exists():
        st.image(str(interpolation_plot), caption="Interpolation over masked test rows", width="stretch")
    elif legacy_plot.exists():
        st.image(str(legacy_plot), caption="Interpolation results", width="stretch")

    if selected_run:
        render_grouped_metric_table(selected_run, "Interpolation Metrics by Store", "interpolation_metrics_by_store.csv")
        render_grouped_metric_table(selected_run, "Interpolation Metrics by Store and Product", "interpolation_metrics_by_store_product.csv")
        render_grouped_metric_table(selected_run, "Interpolation Metrics by Category and Product", "interpolation_metrics_by_category_product.csv")


def render_forecasting() -> None:
    st.header("Forecasting")
    training_runs = list_training_runs()
    selected_run = None
    if training_runs:
        selected_run = st.selectbox("Select trainer run", options=training_runs, format_func=lambda item: item.name, key="forecast_run")
        st.caption(f"Selected trainer run folder: {selected_run.relative_to(ROOT_DIR)}")

    metrics_text = safe_read_text((selected_run / "ml_interpolation_metrics.txt") if selected_run else (OUTPUT_DIR / "ml_interpolation_metrics.txt"))
    report_text = safe_read_text((selected_run / "forecasting_report.md") if selected_run else (OUTPUT_DIR / "forecasting_report.md"))

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Metrics")
        st.code(metrics_text, language="text")
    with col2:
        st.subheader("Report")
        st.markdown(report_text)

    forecast_plot = (selected_run / "daily_forecast_test_horizon.png") if selected_run and (selected_run / "daily_forecast_test_horizon.png").exists() else (PLOTS_DIR / "ml_forecasting" / "daily_forecast_test_horizon.png")
    if forecast_plot.exists():
        st.image(str(forecast_plot), caption="Forecast horizon: Feb-Apr 2026", width="stretch")

    forecast_daily = safe_read_csv((selected_run / "forecast_daily_aggregate.csv") if selected_run else (OUTPUT_DIR / "forecast_daily_aggregate.csv"))
    if not forecast_daily.empty:
        st.subheader("Daily Forecast Aggregate")
        st.line_chart(
            forecast_daily.set_index("date")[["actual_units_sold", "predicted_units_sold"]],
            width="stretch",
        )
        st.dataframe(forecast_daily.tail(90), width="stretch")

    if selected_run:
        render_grouped_metric_table(selected_run, "Forecast Metrics by Store", "forecast_metrics_by_store.csv")
        render_grouped_metric_table(selected_run, "Forecast Metrics by Store and Product", "forecast_metrics_by_store_product.csv")
        render_grouped_metric_table(selected_run, "Forecast Metrics by Category and Product", "forecast_metrics_by_category_product.csv")


def render_logs() -> None:
    st.header("Output Logs")
    log_df = build_log_index()
    if log_df.empty:
        st.warning("No files found in the output folder.")
        return

    timestamps = sorted(log_df["timestamp"].unique(), reverse=True)
    selected_timestamp = st.selectbox("Select log timestamp", timestamps)
    filtered = log_df[log_df["timestamp"] == selected_timestamp]
    st.dataframe(filtered, width="stretch", hide_index=True)

    selected_file = st.selectbox("Preview a file", filtered["relative_path"].tolist())
    preview_path = ROOT_DIR / selected_file
    suffix = preview_path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        st.image(str(preview_path), caption=selected_file, width="stretch")
    elif suffix in {".md", ".txt"}:
        st.code(safe_read_text(preview_path), language="text")
    elif suffix == ".csv":
        st.dataframe(safe_read_csv(preview_path, rows=300), width="stretch")
    else:
        st.info(f"Preview not implemented for {suffix or 'this file type'}. Path: {selected_file}")


def main():
    st.set_page_config(page_title="US Store Sales Forecasting", layout="wide")
    st.sidebar.title("Navigation")
    st.sidebar.caption("Run the app with: `streamlit run src/app.py`")
    page = st.sidebar.radio(
        "Select view",
        ["Overview", "EDA", "Time Series", "Interpolation", "Forecasting", "Logs"],
    )

    if page == "Overview":
        render_overview()
    elif page == "EDA":
        render_eda()
    elif page == "Time Series":
        render_time_series()
    elif page == "Interpolation":
        render_interpolation()
    elif page == "Forecasting":
        render_forecasting()
    else:
        render_logs()


if __name__ == '__main__':
    if get_script_run_ctx() is None:
        print("This is a Streamlit app. Start it with: streamlit run src/app.py")
    else:
        main()
