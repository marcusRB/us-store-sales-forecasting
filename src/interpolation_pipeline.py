import os

import polars as pl

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SQLITE_PATH = os.path.join(DATA_DIR, 'us_store_sales.db')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')

def load_and_join_for_ml():
    sales = pl.read_parquet(os.path.join(DATA_DIR, 'sales.parquet'))
    stores = pl.read_parquet(os.path.join(DATA_DIR, 'stores.parquet'))
    items = pl.read_parquet(os.path.join(DATA_DIR, 'items.parquet'))
    calendar = pl.read_parquet(os.path.join(DATA_DIR, 'calendar.parquet'))
    weather = pl.read_parquet(os.path.join(DATA_DIR, 'weather_events.parquet'))
    macro = pl.read_parquet(os.path.join(DATA_DIR, 'macroeconomic_events.parquet'))
    micro = pl.read_parquet(os.path.join(DATA_DIR, 'microeconomic_events.parquet'))

    df = sales.join(stores, on='store_id', how='left') \
             .join(items, on='item_id', how='left') \
             .join(calendar, on='date', how='left') \
             .join(weather, on=['store_id','date'], how='left')

    for event_column in ['macro_event_id', 'micro_event_id']:
        if event_column in df.columns:
            df = df.with_columns(pl.col(event_column).cast(pl.Int64, strict=False))

    if 'macro_event_id' in df.columns and 'event_id' in macro.columns:
        df = df.join(
            macro.rename({'event_id': 'macro_event_id', 'description': 'macro_event_description'}),
            on='macro_event_id',
            how='left',
        )
    if 'micro_event_id' in df.columns and 'event_id' in micro.columns:
        df = df.join(
            micro.rename({'event_id': 'micro_event_id', 'type': 'micro_event_label'}),
            on='micro_event_id',
            how='left',
        )
    return df


def run_interpolation_pipeline(save_outputs=True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_and_join_for_ml()
    if save_outputs:
        parquet_path = os.path.join(OUTPUT_DIR, 'interpolation_modeling_table.parquet')
        csv_path = os.path.join(OUTPUT_DIR, 'interpolation_modeling_sample.csv')
        df.write_parquet(parquet_path, compression='zstd')
        df.head(1000).write_csv(csv_path)
        print(f'Interpolation modeling table saved to {parquet_path}')
        print(f'Interpolation sample saved to {csv_path}')
    print(f'Interpolation pipeline complete. Joined rows: {df.height}')
    return df

if __name__ == '__main__':
    run_interpolation_pipeline(save_outputs=True)
