import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


np.random.seed(42)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots', 'ml_forecasting')
TRAINING_RUNS_DIR = os.path.join(OUTPUT_DIR, 'training_runs')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TRAINING_RUNS_DIR, exist_ok=True)


def load_data():
    sales = pl.read_parquet(os.path.join(DATA_DIR, 'sales.parquet')).to_pandas()
    stores = pl.read_parquet(os.path.join(DATA_DIR, 'stores.parquet')).to_pandas()
    items = pl.read_parquet(os.path.join(DATA_DIR, 'items.parquet')).to_pandas()
    calendar = pl.read_parquet(os.path.join(DATA_DIR, 'calendar.parquet')).to_pandas()

    df = sales.merge(stores, on='store_id', how='left')
    df = df.merge(items, on='item_id', how='left')
    df = df.merge(calendar[['date', 'quarter', 'is_weekend']], on='date', how='left')

    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

    label_columns = {
        'category': 'category_label',
        'subcategory': 'subcategory_label',
        'holiday_name': 'holiday_name_label',
        'detected_event_type': 'detected_event_type_label',
        'macro_event_type': 'macro_event_type_label',
        'micro_event_type': 'micro_event_type_label',
        'store_type': 'store_type_label',
    }
    for source, target in label_columns.items():
        if source in df.columns:
            df[target] = df[source].fillna('none').astype(str)

    if 'description' in df.columns:
        df['product_label'] = df['description'].fillna('unknown').astype(str)
    else:
        df['product_label'] = df['item_id'].astype(str)

    for column in [
        'store_type', 'category', 'subcategory', 'demand_pattern', 'seasonality_profile',
        'weekly_profile', 'traffic_band', 'detected_event_type', 'macro_event_type',
        'micro_event_type', 'holiday_name'
    ]:
        if column in df.columns:
            df[column] = df[column].fillna('none').astype('category').cat.codes

    for column in ['is_promo_day', 'is_holiday', 'is_stockout', 'is_peak_day', 'is_weekend', 'is_weight_variable', 'is_perishable']:
        if column in df.columns:
            df[column] = df[column].fillna(False).astype(int)

    return df


def split_train_test(df):
    train_end = pd.Timestamp('2026-01-31')
    test_start = pd.Timestamp('2026-02-01')
    test_end = pd.Timestamp('2026-04-30')
    train_df = df[df['date'] <= train_end].copy()
    test_df = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()
    return train_df, test_df


def create_missing_data(df, base_missing_rate=0.08):
    rng = np.random.default_rng(42)
    event_weight = (
        1.0
        + 0.75 * df['is_holiday']
        + 0.65 * df['is_promo_day']
        + 0.9 * (df['macro_event_type'] > 0)
        + 1.2 * (df['micro_event_type'] > 0)
        + 0.8 * df['is_peak_day']
    )
    probabilities = np.clip(base_missing_rate * event_weight, 0.0, 0.35)
    missing_mask = rng.random(len(df)) < probabilities
    df['is_missing'] = missing_mask.astype(int)
    df['units_sold_masked'] = df['units_sold'].copy()
    df.loc[missing_mask, 'units_sold_masked'] = np.nan
    return df


def build_hierarchical_priors(train_df):
    return {
        'store_item_dow': train_df.groupby(['store_id', 'item_id', 'day_of_week'])['units_sold'].mean().to_dict(),
        'store_category_dow': train_df.groupby(['store_id', 'category_label', 'day_of_week'])['units_sold'].mean().to_dict(),
        'category_dow': train_df.groupby(['category_label', 'day_of_week'])['units_sold'].mean().to_dict(),
        'global_dow': train_df.groupby(['day_of_week'])['units_sold'].mean().to_dict(),
    }


def apply_hierarchical_blend(test_df, model_predictions, priors):
    blended = np.zeros(len(test_df), dtype=float)
    for index, row in enumerate(test_df.itertuples(index=False)):
        candidates = [(float(model_predictions[index]), 0.8)]
        store_item = priors['store_item_dow'].get((row.store_id, row.item_id, row.day_of_week))
        if store_item is not None:
            candidates.append((float(store_item), 0.12))
        store_category = priors['store_category_dow'].get((row.store_id, row.category_label, row.day_of_week))
        if store_category is not None:
            candidates.append((float(store_category), 0.05))
        category_day = priors['category_dow'].get((row.category_label, row.day_of_week))
        if category_day is not None:
            candidates.append((float(category_day), 0.02))
        global_day = priors['global_dow'].get(row.day_of_week)
        if global_day is not None:
            candidates.append((float(global_day), 0.01))
        total_weight = sum(weight for _, weight in candidates)
        blended[index] = sum(value * weight for value, weight in candidates) / total_weight
    return np.maximum(blended, 0.0)


def monte_carlo_simulation(test_df, predictions):
    rng = np.random.default_rng(42)
    event_uncertainty = (
        0.14
        + 0.05 * test_df['is_holiday'].to_numpy()
        + 0.045 * test_df['is_promo_day'].to_numpy()
        + 0.1 * (test_df['macro_event_type'].to_numpy() > 0)
        + 0.12 * (test_df['micro_event_type'].to_numpy() > 0)
        + 0.08 * test_df['is_peak_day'].to_numpy()
        + 0.02 * np.maximum(0, test_df['inflation_rate'].to_numpy() - 3.0)
    )
    zero_event_prob = np.clip(
        0.01
        + 0.03 * (test_df['macro_event_type'].to_numpy() > 0)
        + 0.08 * (test_df['micro_event_type'].to_numpy() > 0)
        + 0.02 * test_df['is_holiday'].to_numpy() * (test_df['holiday_name'].to_numpy() == 0),
        0.0,
        0.4,
    )
    scenario_count = 300
    simulations = np.zeros((len(test_df), scenario_count))

    for scenario in range(scenario_count):
        lambda_draw = np.maximum(predictions, 0.05) * rng.lognormal(mean=0.0, sigma=event_uncertainty)
        collapse_draw = rng.random(len(test_df)) < zero_event_prob
        poisson_draw = rng.poisson(lambda_draw)
        lumpy_mask = test_df['demand_pattern'].to_numpy() > 1
        if lumpy_mask.any():
            dispersion = np.maximum(test_df.loc[lumpy_mask, 'overdispersion'].to_numpy(), 1.0)
            lumpy_lambda = np.maximum(lambda_draw[lumpy_mask], 0.05)
            probability = dispersion / (dispersion + lumpy_lambda)
            poisson_draw[lumpy_mask] = rng.negative_binomial(dispersion, probability)
        poisson_draw[collapse_draw] = 0
        simulations[:, scenario] = poisson_draw

    return simulations


def metric_dict(actual, predicted, prefix=''):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    abs_error = np.abs(actual - predicted)
    denom_abs_actual = np.abs(actual).sum()
    non_zero_mask = np.abs(actual) > 1e-8

    metrics = {
        f'{prefix}count': int(len(actual)),
        f'{prefix}actual_sum': float(actual.sum()),
        f'{prefix}predicted_sum': float(predicted.sum()),
        f'{prefix}mae': float(mean_absolute_error(actual, predicted)),
        f'{prefix}rmse': float(np.sqrt(mean_squared_error(actual, predicted))),
        f'{prefix}bias': float((predicted - actual).mean()),
        f'{prefix}mbe': float((predicted - actual).sum()),
        f'{prefix}wape': float(abs_error.sum() / denom_abs_actual) if denom_abs_actual > 0 else np.nan,
        f'{prefix}smape': float(np.mean(2 * abs_error / np.maximum(np.abs(actual) + np.abs(predicted), 1e-8))),
    }
    metrics[f'{prefix}mape'] = float(np.mean(abs_error[non_zero_mask] / np.abs(actual[non_zero_mask]))) if non_zero_mask.any() else np.nan
    try:
        metrics[f'{prefix}r2'] = float(r2_score(actual, predicted))
    except ValueError:
        metrics[f'{prefix}r2'] = np.nan
    return metrics


def compute_group_metrics(dataframe, group_cols, actual_col, prediction_col, min_count=20):
    rows = []
    for keys, group in dataframe.groupby(group_cols):
        if len(group) < min_count:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(group_cols, keys)}
        row.update(metric_dict(group[actual_col], group[prediction_col]))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(['wape', 'rmse', 'mae', 'count'], ascending=[True, True, True, False])


def save_metrics(metrics, run_dir):
    for path in [os.path.join(OUTPUT_DIR, 'ml_interpolation_metrics.txt'), os.path.join(run_dir, 'ml_interpolation_metrics.txt')]:
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write('ML Forecasting, Event-Aware Interpolation and Monte Carlo Simulation\n')
            handle.write('==============================================================\n')
            for key, value in metrics.items():
                handle.write(f'{key}: {value}\n')


def plot_results(run_dir, daily_forecast, masked_eval):
    plt.figure(figsize=(14, 6))
    plt.plot(daily_forecast['date'], daily_forecast['actual_units_sold'], label='Actual daily sales', linewidth=2)
    plt.plot(daily_forecast['date'], daily_forecast['predicted_units_sold'], label='Predicted daily sales', linewidth=2)
    plt.fill_between(daily_forecast['date'], daily_forecast['p05_units_sold'], daily_forecast['p95_units_sold'], alpha=0.25, label='Monte Carlo 90% interval')
    plt.title('Forecasting on Feb-Apr 2026 Test Horizon')
    plt.xlabel('Date')
    plt.ylabel('Units sold')
    plt.legend()
    plt.tight_layout()
    for path in [os.path.join(PLOTS_DIR, 'daily_forecast_test_horizon.png'), os.path.join(run_dir, 'daily_forecast_test_horizon.png')]:
        plt.savefig(path)
    plt.close()

    subset_size = min(150, len(masked_eval))
    if subset_size > 0:
        subset = masked_eval.head(subset_size)
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(subset_size), subset['units_sold'], label='True masked units', marker='o')
        plt.plot(np.arange(subset_size), subset['interpolated_units'], label='Interpolated mean', marker='x')
        plt.fill_between(np.arange(subset_size), subset['p05'], subset['p95'], alpha=0.25, label='Monte Carlo 90% interval')
        plt.title('Interpolation on Artificially Missing Test Rows')
        plt.xlabel('Masked sample index')
        plt.ylabel('Units sold')
        plt.legend()
        plt.tight_layout()
        for path in [os.path.join(PLOTS_DIR, 'interpolation_masked_test_rows.png'), os.path.join(run_dir, 'interpolation_masked_test_rows.png')]:
            plt.savefig(path)
        plt.close()


def save_grouped_metrics(run_dir, grouped_tables):
    for name, dataframe in grouped_tables.items():
        if dataframe.empty:
            continue
        dataframe.to_csv(os.path.join(OUTPUT_DIR, f'{name}.csv'), index=False)
        dataframe.to_csv(os.path.join(run_dir, f'{name}.csv'), index=False)


def save_run_manifest(run_dir, metrics, grouped_tables):
    manifest = {
        'run_timestamp': os.path.basename(run_dir),
        'metrics': metrics,
        'grouped_tables': sorted([name for name, dataframe in grouped_tables.items() if not dataframe.empty]),
    }
    with open(os.path.join(run_dir, 'manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)


def run_ml():
    print('Running event-aware ML forecasting and Monte Carlo interpolation...')
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(TRAINING_RUNS_DIR, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)

    df = load_data()
    train_df, test_df = split_train_test(df)
    test_df = create_missing_data(test_df)
    print(f'Training rows: {len(train_df)}')
    print(f'Test rows (Feb-Apr 2026): {len(test_df)}')

    features = [
        'store_id', 'item_id', 'day_of_week', 'week_of_year', 'month', 'quarter', 'year',
        'is_weekend', 'is_month_start', 'is_month_end', 'base_price', 'size_sqft',
        'avg_customer_traffic', 'price_sensitivity', 'holiday_uplift', 'elasticity',
        'avg_daily_demand', 'popularity_score', 'zero_inflation_prob', 'overdispersion',
        'weight_kg_per_unit', 'is_weight_variable', 'is_perishable', 'category', 'subcategory',
        'demand_pattern', 'seasonality_profile', 'weekly_profile', 'store_type', 'traffic_band',
        'is_promo_day', 'is_holiday', 'is_peak_day', 'macro_event_type', 'micro_event_type',
        'holiday_name', 'inflation_rate', 'regional_unemployment_rate', 'temp_celsius', 'precipitation_mm'
    ]
    features = [column for column in features if column in df.columns]
    X_train = train_df[features].fillna(0)
    y_train = train_df['units_sold']
    X_test = test_df[features].fillna(0)
    y_test = test_df['units_sold']

    print('Training HistGradientBoostingRegressor...')
    model = HistGradientBoostingRegressor(random_state=42, max_iter=220, max_depth=8, learning_rate=0.05)
    model.fit(X_train, y_train)

    model_pred = np.maximum(model.predict(X_test), 0)
    priors = build_hierarchical_priors(train_df)
    blended_pred = apply_hierarchical_blend(test_df, model_pred, priors)
    simulations = monte_carlo_simulation(test_df, blended_pred)

    test_df['model_predicted_units'] = model_pred
    test_df['predicted_units'] = blended_pred
    test_df['scenario_mean'] = simulations.mean(axis=1)
    test_df['scenario_median'] = np.median(simulations, axis=1)
    test_df['scenario_p05'] = np.percentile(simulations, 5, axis=1)
    test_df['scenario_p95'] = np.percentile(simulations, 95, axis=1)
    test_df['within_interval'] = ((test_df['units_sold'] >= test_df['scenario_p05']) & (test_df['units_sold'] <= test_df['scenario_p95'])).astype(int)

    masked_eval = test_df[test_df['is_missing'] == 1].copy()
    masked_eval['interpolated_units'] = masked_eval['scenario_mean']
    masked_eval['p05'] = masked_eval['scenario_p05']
    masked_eval['p95'] = masked_eval['scenario_p95']

    forecast_metrics = metric_dict(y_test, blended_pred, prefix='forecast_')
    coverage_90 = float(test_df['within_interval'].mean())
    if len(masked_eval) > 0:
        interpolation_metrics = metric_dict(masked_eval['units_sold'], masked_eval['interpolated_units'], prefix='interpolation_')
    else:
        interpolation_metrics = {key: np.nan for key in [
            'interpolation_count', 'interpolation_actual_sum', 'interpolation_predicted_sum', 'interpolation_mae',
            'interpolation_rmse', 'interpolation_bias', 'interpolation_mbe', 'interpolation_wape',
            'interpolation_smape', 'interpolation_mape', 'interpolation_r2'
        ]}

    daily_forecast = (
        test_df.groupby('date', as_index=False)
        .agg(
            actual_units_sold=('units_sold', 'sum'),
            predicted_units_sold=('predicted_units', 'sum'),
            p05_units_sold=('scenario_p05', 'sum'),
            p95_units_sold=('scenario_p95', 'sum'),
        )
        .sort_values('date')
    )

    grouped_tables = {
        'forecast_metrics_by_store': compute_group_metrics(test_df, ['store_id'], 'units_sold', 'predicted_units', min_count=40),
        'forecast_metrics_by_store_product': compute_group_metrics(test_df, ['store_id', 'item_id', 'product_label'], 'units_sold', 'predicted_units', min_count=12),
        'forecast_metrics_by_category_product': compute_group_metrics(test_df, ['category_label', 'item_id', 'product_label'], 'units_sold', 'predicted_units', min_count=20),
        'interpolation_metrics_by_store': compute_group_metrics(masked_eval, ['store_id'], 'units_sold', 'interpolated_units', min_count=20),
        'interpolation_metrics_by_store_product': compute_group_metrics(masked_eval, ['store_id', 'item_id', 'product_label'], 'units_sold', 'interpolated_units', min_count=8),
        'interpolation_metrics_by_category_product': compute_group_metrics(masked_eval, ['category_label', 'item_id', 'product_label'], 'units_sold', 'interpolated_units', min_count=10),
    }

    for dataframe, latest_path, run_path in [
        (test_df, os.path.join(OUTPUT_DIR, 'forecast_test_rows.csv'), os.path.join(run_dir, 'forecast_test_rows.csv')),
        (masked_eval, os.path.join(OUTPUT_DIR, 'interpolation_masked_rows.csv'), os.path.join(run_dir, 'interpolation_masked_rows.csv')),
        (daily_forecast, os.path.join(OUTPUT_DIR, 'forecast_daily_aggregate.csv'), os.path.join(run_dir, 'forecast_daily_aggregate.csv')),
    ]:
        dataframe.to_csv(latest_path, index=False)
        dataframe.to_csv(run_path, index=False)

    metrics = {
        'run_timestamp': run_timestamp,
        'train_horizon_end': '2026-01-31',
        'test_horizon_start': '2026-02-01',
        'test_horizon_end': '2026-04-30',
        'coverage_90_interval': round(coverage_90, 4),
        'masked_rows': int(len(masked_eval)),
        'test_rows': int(len(test_df)),
        **{key: (round(value, 4) if isinstance(value, float) and not np.isnan(value) else value) for key, value in forecast_metrics.items()},
        **{key: (round(value, 4) if isinstance(value, float) and not np.isnan(value) else value) for key, value in interpolation_metrics.items()},
    }
    save_metrics(metrics, run_dir)
    save_grouped_metrics(run_dir, grouped_tables)
    plot_results(run_dir, daily_forecast, masked_eval)
    save_run_manifest(run_dir, metrics, grouped_tables)

    best_sections = [(name, dataframe.head(10)) for name, dataframe in grouped_tables.items() if not dataframe.empty]
    for path in [os.path.join(OUTPUT_DIR, 'forecasting_report.md'), os.path.join(run_dir, 'forecasting_report.md')]:
        with open(path, 'w', encoding='utf-8') as report:
            report.write('# Event-Aware Forecasting and Monte Carlo Simulation\n\n')
            report.write('The forecasting pipeline trains on 24 months of synthetic history and evaluates on February to April 2026. The interpolation layer blends model demand with hierarchical priors from store-item, store-category, and category-day patterns before Monte Carlo simulation.\n\n')
            report.write('## Global Metrics\n')
            for key, value in metrics.items():
                report.write(f'- {key}: {value}\n')
            report.write('\n## Best Combination Tables\n')
            for name, table in best_sections:
                report.write(f'\n### {name}\n')
                report.write(table.to_csv(index=False))
                report.write('\n')

    print('Forecasting completed.')
    print(f"Best forecast-by-store WAPE: {grouped_tables['forecast_metrics_by_store']['wape'].iloc[0]:.4f}" if not grouped_tables['forecast_metrics_by_store'].empty else 'Best forecast-by-store WAPE: n/a')
    print(f"Best interpolation-by-store-product WAPE: {grouped_tables['interpolation_metrics_by_store_product']['wape'].iloc[0]:.4f}" if not grouped_tables['interpolation_metrics_by_store_product'].empty else 'Best interpolation-by-store-product WAPE: n/a')
    print(f'Outputs saved in {OUTPUT_DIR}, {PLOTS_DIR}, and {run_dir}')


if __name__ == '__main__':
    run_ml()
