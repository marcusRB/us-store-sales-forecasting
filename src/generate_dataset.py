import polars as pl
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ---------- Config ----------
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2026, 4, 30)
N_STORES = 100
N_ITEMS = 1000
SEED = 42
np.random.seed(SEED)

# ---------- 1. Calendar ----------
dates = pl.date_range(START_DATE, END_DATE, interval='1d', eager=True)
df_cal = pl.DataFrame({'date': dates})
df_cal = df_cal.with_columns([
    pl.col('date').dt.weekday().alias('day_of_week'),  # Monday=1... Sunday=7
    pl.col('date').dt.week().alias('week'),
    pl.col('date').dt.month().alias('month'),
    pl.col('date').dt.year().alias('year'),
])
# Add holiday flags later after holidays_events

# ---------- 2. Stores ----------
# Generate 100 stores with clusters
store_clusters = np.random.choice([0,1,2,3,4], size=N_STORES, p=[0.3,0.2,0.2,0.15,0.15])
states = ['CA','TX','NY','FL','IL','PA','OH','GA','NC','MI']  # example
regions = {'CA':'West','TX':'South','NY':'Northeast', ...} # mapping
# Build store dataframe
df_stores = pl.DataFrame({
    'store_id': range(1, N_STORES+1),
    'cluster_id': store_clusters,
    'state': np.random.choice(states, N_STORES),
    # ... other columns
})
# assign weather_zone and unemployment_region based on state/cluster

# ---------- 3. Items ----------
categories = ['Grocery','Health & Beauty','Household Essentials']
# further subdivision...
# create 1000 items with distributions for base_price, weight, demand_pattern
df_items = pl.DataFrame({
    'item_id': range(1, N_ITEMS+1),
    # ...
})

# ---------- 4. External factors ----------
# Generate monthly economic series (we'll expand to daily later)
months = pl.date_range(START_DATE.replace(day=1), END_DATE.replace(day=1), '1mo', eager=True)
df_oil = pl.DataFrame({'date': months, 'oil_price_usd': random_oil_walk(len(months))})
df_inflation = pl.DataFrame({'date': months, 'cpi_yoy': random_inflation(len(months))})
# unemployment per region: create one series per region_id
# etc.

# ---------- 5. Weather ----------
# For each store, generate daily weather using sine + noise. 
# We'll build a long dataframe: store_id, date, temp, precip, flags.

# ---------- 6. Events & holidays ----------
df_holidays = create_holiday_calendar(START_DATE, END_DATE)  # returns (date, holiday_name, impact, affected_states)
df_macro = create_macro_events()
df_micro = create_micro_events(df_stores, START_DATE, END_DATE)

# ---------- 7. Promotions ----------
df_promo = generate_promotions(df_stores, df_items, START_DATE, END_DATE)

# ---------- 8. Demand & Sales simulation ----------
# This is the heavy part. We will compute a huge table, but optimized:
# 1. Create a full combination of (store_id, item_id) that have average demand > 0 (sparsity).
# 2. For each day in the range, compute demand factors and draw sales.
# We'll use Polars' lazy API and maybe groupby over date to limit memory.

# Define a function that, given store_ids and item_ids, produces one DataFrame with columns:
# store_id, item_id, date, units_sold, weight_kg, price_per_unit, promo_id, is_stockout, true_demand

# We'll use vectorized operations and maybe generate in chunks per month.

# Save as Parquet partitioned by year/month or by store.
df_sales.write_parquet('sales.parquet', compression='zstd')