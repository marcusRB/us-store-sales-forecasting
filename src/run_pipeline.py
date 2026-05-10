import argparse
import sys

from clustering import run_clustering
from eda import run_eda
from interpolation_pipeline import run_interpolation_pipeline
from ml_forecasting import run_ml
from time_series_anomalies import analyze_ts_and_anomalies


STEP_ALIASES = {
    'eda': 'eda',
    'analysis': 'eda',
    'time-series': 'time-series',
    'timeseries': 'time-series',
    'anomalies': 'anomalies',
    'forecasting': 'forecasting',
    'forecast': 'forecasting',
    'interpolation': 'interpolation',
    'clustering': 'clustering',
    'all': 'all',
}

STEP_RUNNERS = {
    'eda': run_eda,
    'time-series': analyze_ts_and_anomalies,
    'anomalies': analyze_ts_and_anomalies,
    'clustering': run_clustering,
    'interpolation': run_interpolation_pipeline,
    'forecasting': run_ml,
}

ORDERED_STEPS = ['eda', 'time-series', 'clustering', 'interpolation', 'forecasting']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run one or more pipeline stages for the US store sales forecasting project.'
    )
    parser.add_argument(
        '--steps',
        nargs='+',
        default=['all'],
        help='Pipeline steps to run: eda, time-series, anomalies, clustering, interpolation, forecasting, all.',
    )
    return parser.parse_args()


def normalize_steps(raw_steps):
    normalized = []
    for raw_step in raw_steps:
        key = STEP_ALIASES.get(raw_step.strip().lower())
        if key is None:
            valid_steps = ', '.join(sorted(STEP_ALIASES.keys()))
            raise ValueError(f'Unknown step: {raw_step}. Valid values: {valid_steps}')
        if key == 'all':
            normalized.extend(ORDERED_STEPS)
        else:
            normalized.append(key)

    unique_steps = []
    for step in ORDERED_STEPS:
        if step in normalized and step not in unique_steps:
            unique_steps.append(step)
    return unique_steps


def main():
    args = parse_args()
    try:
        steps = normalize_steps(args.steps)
    except ValueError as exc:
        print(str(exc))
        return 2

    print(f'Running pipeline steps: {", ".join(steps)}')
    for step in steps:
        print(f'\n=== {step} ===')
        STEP_RUNNERS[step]()

    print('\nPipeline execution completed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())