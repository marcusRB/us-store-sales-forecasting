# US Store sales Forecasting

We’ll design a full data‑generation pipeline that creates a realistic, multi‑table retail dataset for your sales forecasting MVP. The approach uses **latent clusters** for stores and products so that you can train only a handful of models, not one per store. All external factors (weather, holidays, economics, promotions, stock‑outs) are simulated with plausible real‑world dynamics.

The output will be a set of **Parquet files** (efficient, columnar) and, optionally, a **SQLite database** — both produced by a single Python script. We’ll use **Polars** for speed and memory efficiency, with Dask as an alternative if you want to scale even further.

---

## 1. Overview of the Dataset Ecosystem

We will create **11 base tables** that fully describe the retail environment. They can be joined on common keys (`store_id`, `item_id`, `date`) to form a denormalised fact table for modelling.

| Table | Description |
|-------|-------------|
| `stores` | 100 stores with geographic, demographic, and cluster features |
| `items` | 1000 SKUs with category, base price, weight, demand pattern |
| `calendar` | Date dimension with day‑of‑week, month, year, and a holiday flag |
| `holidays_events` | National & regional holidays, festivals, with impact factors |
| `weather_events` | Daily weather per store (temperature, precipitation, extreme flags) |
| `oil_price` | Monthly oil price (proxy for transport / input costs) |
| `inflation_rate` | Monthly CPI inflation |
| `unemployment_rate` | Monthly unemployment by state / region |
| `interest_fed_rate` | Federal funds rate (quarterly changes) |
| `macroeconomic_events` | Rare nationwide shocks (recession indicator, policy change) |
| `microeconomic_events` | Local disruptions: strike, road closure, store renovation |
| `promotions` | Planned price discounts (%, BOGO, bulk) per store–item–date |
| `sales` | Daily units sold & weight (kg), final price, discount applied |

You can then build a star schema: `sales` is the fact table; `stores`, `items`, `calendar` are dimensions; the rest are slowly‑changing or event dimensions.

---

## 2. Schema Design (Key Columns)

I’ll outline the critical columns for each table; the full SQL can be auto‑generated from the Polars DataFrames.

### `stores`
- `store_id`: int (1…100)
- `store_name`, `city`, `county`, `state`, `region` (Northeast, Midwest, South, West)
- `store_type` (Urban, Suburban, Rural)
- `size_sqft`, `avg_customer_traffic`
- `cluster_id`: latent cluster 0…4 (captures overall demand pattern, price sensitivity, etc.)
- `unemployment_region_id` to link to regional unemployment series
- `weather_zone_id` to link to weather grid

### `items`
- `item_id`: int (1…1000)
- `sku_code` (e.g., “FRESH-001”, “BEAUTY-102”)
- `description`: short text (e.g., “Organic Whole Chicken 1.5kg”)
- `category`: `Grocery`, `Health & Beauty`, `Household Essentials`
- `subcategory`: `Fresh Food`, `Other Food`, `Generic`, `Personal Care`, etc.
- `base_price` (PVP in USD)
- `weight_kg_per_unit` (mean, with a std for variability)
- `is_weight_variable` (True for meat, bulk items)
- `is_perishable` (affects sales frequency and stock‑outs)
- `demand_pattern`: `Regular`, `Sporadic`, `Lumpy` (controls probability of zero sales)
- `cluster_id`: latent cluster 0…9 capturing price elasticity, seasonality shape, promotion uplift

### `calendar`
- `date` (2024‑01‑01 to 2026‑04‑30), `day_of_week`, `week`, `month`, `quarter`, `year`
- `is_holiday`, `holiday_id` (foreign key to `holidays_events`)

### `holidays_events`
- `holiday_id`, `holiday_name`, `date`, `is_national` (True/False), `affected_states` (list or NULL)
- `impact_factor`: multiplier on baseline demand (e.g., 0.3 for a quiet day, 1.8 for Christmas)
- `type`: `Public`, `Regional`, `Religious`, `School Vacation`

### `weather_events`
- `store_id`, `date`
- `temp_celsius` (daily mean), `precipitation_mm`, `snowfall_mm`
- `is_extreme_heat`, `is_extreme_cold`, `is_storm` (flags derived from thresholds)
- Derived from a seasonal stochastic model per weather zone.

### Economic Indicators (monthly resolution)
**`oil_price`**: `date` (1st of month), `price_usd_per_barrel`  
**`inflation_rate`**: `date`, `cpi_yoy_pct`  
**`unemployment_rate`**: `date`, `region_id`, `rate_pct`  
**`interest_fed_rate`**: `date`, `fed_funds_rate_pct`

### `macroeconomic_events`
- `event_id`, `date`, `description` (e.g., “Recession warning”, “Stimulus check”), `demand_multiplier` (applied nationwide for a window)

### `microeconomic_events`
- `event_id`, `store_id`, `start_date`, `end_date`, `type` (Strike, Renovation, Road Closure), `demand_multiplier`

### `promotions`
- `promotion_id`, `store_id`, `item_id`, `start_date`, `end_date`
- `discount_type` (`percentage_off`, `bogo`, `bulk_discount`, `weight_step` – e.g., 3kg pack 20% cheaper per kg)
- `discount_value` (e.g., 0.15 for 15% off)
- `promo_multiplier` (how much does demand increase under this promotion? can be estimated from `discount_value` and item price elasticity)

### `sales` (daily grain)
- `store_id`, `item_id`, `date`
- `units_sold` (integer, can be 0 but we flag stock‑outs separately)
- `weight_kg_sold` (if weight‑variable, otherwise `units_sold * avg_weight`)
- `price_per_unit_after_discount` (actual price paid per unit)
- `promotion_id` (nullable)
- `is_stockout` (Boolean) – zero sales because shelf empty, not because lack of demand
- `base_demand` (hidden ground‑truth demand for evaluation / interpolation)

---

## 3. Generation Methodology Step by Step

We’ll write a single Python script using **Polars** (for speed) and **NumPy/SciPy** for random generation.

### 3.1 Stores and Clusters

- Generate 100 stores with random geographic distribution across 10 states, weighted by population.
- Assign `cluster_id` using a mixture model: pick from 5 cluster centres (latent type) each defining:
  - Temperature zone (1–4)
  - Typical price sensitivity (low/medium/high)
  - Base traffic volume (low/medium/high)
  - Holiday uplift factor
- Link each store to an unemployment region and weather zone.

### 3.2 Items and Demand Patterns

- Create 1000 items divided among categories:  
  - 400 Grocery (200 Fresh Food, 200 Other Food)  
  - 300 Health & Beauty  
  - 300 Household Essentials  
- Assign subcategories, realistic base prices, and weight ranges (e.g., fresh meat 0.5‑3 kg, beauty products 0.1‑0.5 kg, detergents 1‑5 L/kg).
- Each item gets a `demand_pattern` label drawn from:
  - `Regular`: sold almost every day (milk, bread) → Poisson with high mean.
  - `Sporadic`: sold many days but with zeros (specialty sauce) → Zero‑inflated Poisson.
  - `Lumpy`: rare large sales (seasonal chocolates) → Negative Binomial with low probability.
- Assign each item to one of 10 latent product clusters that share:
  - Weekly seasonality curve (e.g., weekend peak vs. mid‑week)
  - Yearly seasonality (summer peak, December peak, none)
  - Price elasticity factor (how much demand changes with price/discount)
  - Promo uplifts for different discount types.

### 3.3 Calendar and Holidays

- Use `polars.date_range` for 2024‑01‑01 to 2026‑04‑30.
- Mark weekends, US federal holidays, and add regional holidays (e.g., Mardi Gras for Louisiana, Patriots’ Day for Massachusetts).  
- Assign `impact_factor` to each holiday based on typical retail behaviour.

### 3.4 Economic & Event Series

- Simulate monthly economic series using a VAR model (or simpler autoregressive process) to maintain realistic correlations:
  - Inflation slowly moves up, then stabilises.
  - Unemployment varies by region with a slow seasonal component.
  - Oil price follows a random walk with reverting shocks.
  - Fed rate changes gradually.
- Macro events: randomly inject 2–3 events over two years (e.g., “consumer confidence drop” with multiplier 0.8 for 4 weeks).
- Micro events: for each store, draw a few interruption events (renovations, strikes) lasting 3–14 days, multiplicatively reducing sales to 0 or 0.2.

### 3.5 Weather Simulation

- For each weather zone, model temperature as a sine wave plus autocorrelated noise; precipitation as a gamma distribution whose mean varies seasonally.
- Define extreme thresholds (e.g., >35°C heat wave, flooding rain) that trigger a flag and affect sales (e.g., reduce foot traffic by 20%).

### 3.6 Promotions Engine

- For each store, create a promotion calendar:
  - Pick random items and schedule 1–3 promotions per month.
  - Discount types: straightforward `percentage_off` (10‑40%), `BOGO`, `bulk_discount` (e.g., buy 3kg pack get 25% discount per kg vs 1kg).
  - For weight‑variable items, we might simulate “buy 3kg for $X” which changes the effective price per kg.
- The `promo_multiplier` for each promo is computed as:
  `uplift = 1 + elasticity_item * discount_rate * base_effectiveness`  
  plus a store‑level modifier.

### 3.7 Demand & Sales Generation (Realistic)

This is the core. We want daily store‑item sales that reflect all the above influences.

**Base demand** `D_base` for store `s`, item `i`, day `t`:
```
D_base(s,i,t) = D_avg(s,i) * seasonality_day_of_week(t) * seasonality_year(t) 
               * trend(t) * store_cluster_factor * item_cluster_factor
```
`D_avg(s,i)` is the long‑term average daily units, drawn from log‑normal distribution based on store size, item popularity, and whether it is a “regular” product in that store (many store‑item pairs will have zero average demand: not all products are sold in all stores). This creates sparsity.

**Adjusted demand** `D_adj(s,i,t)`:
```
D_adj = D_base * holiday_factor(t) * temperature_effect(temp) * precip_effect(precip) 
        * macro_event_multiplier * micro_event_multiplier * price_elasticity_effect(base_price, discounted_price)
```

Where price elasticity effect:
```
if discounted_price < base_price:
    price_factor = (base_price / discounted_price) ** (-elasticity)
else:
    price_factor = (base_price / discounted_price) ** (-elasticity)   # same formula, elasticity < 0
```
The item’s `elasticity` is negative (e.g., -1.5), so lower price increases demand.

**Actual units sold** `units_sold` is then drawn from a distribution matching the `demand_pattern`:
- Regular: `Poisson(D_adj)`
- Sporadic: Zero‑inflated Poisson (with probability `p_zero` that day is a no‑sale day, else Poisson(D_adj))
- Lumpy: Negative Binomial with mean `D_adj` and overdispersion factor.

For weight‑variable items, we also simulate actual weight per unit sold. Each unit’s weight is drawn from `Normal(mean_weight, std_weight)` and then aggregated per transaction (but we keep daily level: total weight = sum of individual unit weights). We can approximate by drawing total weight from a Gamma distribution with mean `units_sold * mean_weight` and variance proportional.

**Stock‑out simulation**: A stock‑out makes `units_sold = 0` regardless of demand, and we set `is_stockout = True`. The probability of stock‑out on a given day depends on:
- Perishability of the item (fresh food has higher stock‑out risk if not replenished daily)
- Recent sales surge (depletes inventory)
- Whether yesterday was a delivery day (we can model deliveries implicitly: after a stock‑out, assume immediate restocking next day if not holiday)

We’ll include a simple “inventory level” model to make stock‑outs realistic, but for MVP we can use a random process: each day, `P(stockout) = base_risk * (1 + 0.5 * yesterday_had_stockout)`, with `base_risk` around 2‑5%.

**Record generation**: Only rows where `D_adj > 0` (or we force zero sales for days with events) are written to the `sales` table, plus we can optionally output a “zero‑sale” flag for interpolation tasks. This keeps the table sparse (typically ~20‑40% of store‑item‑day combos have non‑zero sales).

### 3.8 Missing Days and Monte Carlo Interpolation

To make the dataset useful for testing interpolation methods, we deliberately delete some sales records (e.g., 5% of days at random) and mark them as `missing` in a separate column. These gaps can then be imputed using Monte Carlo simulation of the adjusted demand distribution. Your referenced papers use exactly this approach: simulate many possible demand paths, then average.

We’ll keep a `sales` table with `units_sold`, but we also save a `true_demand` column (the expected demand `D_adj` before stock‑out and before artificial gaps). The final dataset for modelling will have `units_sold` with NaN for missing days and 0 for genuine zero sales. This allows you to evaluate forecasting and interpolation accuracy.

---

## 4. Implementation with Polars (Code Skeleton)

Below is a high‑level script structure. All data is generated as Polars DataFrames and written to Parquet.

```python
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
```

Detailed implementations for each block are beyond a single message, but the full script would be modular and configurable.

**Performance considerations**:  
- With Polars, a full month of data (100 stores * 1000 items * 30 days = 3M rows) is generated in seconds if vectorised.  
- We can keep only rows where `units_sold > 0` or `true_demand > threshold`, reducing size to ~1M per month.  
- If you need all zero‑sales days for interpolation, you can keep them; the total over 28 months would be ~100*1000*850 ≈ 85M rows. That’s manageable in Parquet with good compression (zstd). Use `pl.scan_parquet` for lazy loading.

---

## 5. Building the Final Dataset

From the Parquet files, you can assemble the modelling dataset. A recommended approach:

1. Load `sales` as fact table.
2. Join `stores`, `items`, `calendar` as dimensions.
3. Join `weather_events` by `store_id` and `date`.
4. For daily economic data, you may have monthly values; merge them by joining on year‑month (derived from date) and region where applicable.
5. Join `promotions` on `store_id`, `item_id`, `date` (or carry forward promo_id for active promotion days).
6. Add flags from `holidays_events` via `calendar.holiday_id`.
7. Add micro/macro event multipliers by joining on event windows.

This yields a wide denormalised table with all features, ready for ML. You can store this as another Parquet file or a SQLite DB.

To generate a SQLite database from the script, you can iterate over the Polars DataFrames and use `to_pandas().to_sql()` or dump to SQL using Python’s sqlite3. For large tables like sales, it’s better to keep them as Parquet and only load a sample into SQLite for exploration.

---

## 6. Clustering Strategy for Modelling

The dataset is engineered with latent clusters so you can validate that a few models work well:
- **Store clustering**: use store features (size, type, region, weather zone) or sales profiles (total weekly sales, category mix). You aim for 5–10 clusters.
- **Product clustering**: use demand pattern, category, seasonality, price elasticity. Aim for 10–20 clusters.
- Then train a single forecasting model per cluster‑pair, or a multi‑output model conditioned on cluster id.

Because the true clusters are known (the `cluster_id` columns), you can test if those latent clusters indeed capture most variance.

---

## 7. Monte Carlo Interpolation of Missing Days

Your dataset will include artificial gaps (missing `units_sold`). To interpolate:
1. Fit a probabilistic demand model (e.g., Poisson distribution with parameters depending on features) to the non‑missing data.
2. For each missing day, draw `N` random samples from the predicted distribution (Monte Carlo).
3. The interpolated value can be the mean of these samples (or median), and you get a confidence interval.

The `true_demand` column allows you to measure interpolation error.

---

## 8. Summary and Next Steps

I recommend building this in stages:
1. First, create the static dimension tables (stores, items, calendar).
2. Simulate external series and events.
3. Implement the sales engine and output Parquet.
4. Validate realism by plotting sample store sales.
5. Create the denormalised view for modelling.

All code should be in Python, heavily using Polars. You will end up with a data lake of Parquet files that serve as a gold standard for your forecasting and clustering experiments. The dataset will include all the required tables, realistic promotions, external shocks, and missing data patterns needed for a robust MVP.


---

## 9. How to Generate and Use the Dataset

### Prerequisites
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- If you are using the project conda environment, prefer running all commands with `conda run -n py311-sales ...`.

### Step 1: Generate the Dataset
Run the following script to generate all tables as Parquet files (in `src/data/`) and as a SQLite database (`src/data/us_store_sales.db`):

```bash
conda run -n py311-sales python src/generate_full_dataset.py
```

This version of the generator is event-driven. It now produces:

- explicit holiday effects
- store-item promotions with elasticity-based uplift
- macro stress and rebound windows derived from inflation, oil, and rates
- store-level micro disruptions such as renovation, road closure, outage, and inventory-system failures
- zero-sale events with specific causes instead of unlabeled zeros
- event-aware peak indicators for anomaly detection and forecasting

### Step 2: Running Pipeline Analysis

Once data is generated, you can run the whole analysis pipeline from a single entrypoint:

```bash
conda run -n py311-sales python src/run_pipeline.py --steps all
```

You can also run individual stages or combine only the ones you need:

1. **EDA & Statistics**:
   ```bash
  conda run -n py311-sales python src/run_pipeline.py --steps eda
   ```
2. **Time Series & Anomalies**:
   ```bash
  conda run -n py311-sales python src/run_pipeline.py --steps time-series
   ```
3. **Clustering**:
   ```bash
  conda run -n py311-sales python src/run_pipeline.py --steps clustering
   ```
 4. **Interpolation Modeling Table**:
   ```bash
  conda run -n py311-sales python src/run_pipeline.py --steps interpolation
  ```
 5. **ML Forecasting & Montecarlo Interpolation**:
  ```bash
  conda run -n py311-sales python src/run_pipeline.py --steps forecasting
   ```

Example combined run:

```bash
conda run -n py311-sales python src/run_pipeline.py --steps eda time-series interpolation forecasting
```

### Step 3: Event-Aware Outputs

The anomaly script now produces event-attributed outputs in `output/analysis/` and `output/plots/time_series/`.

- `daily_event_summary.csv`: aggregated daily sales, zero-sale ratios, event counts, and rolling anomaly scores
- `daily_anomalies.csv`: detected anomalies with a causal label such as `holiday_peak`, `promotion_peak`, `macro_stress_drop`, `micro_disruption`, `zero_sale_cluster`, or `peak_surge`
- `anomaly_report.md`: a compact written explanation of the anomaly logic and summary counts

The forecasting script now uses a fixed horizon:

- training data: January 2024 to January 2026
- test data: February, March, and April 2026

It saves:

- row-level predictions for the test horizon
- Monte Carlo scenario intervals
- interpolation results for artificially masked rows
- daily aggregated forecast curves and plots

### Step 4: View Results in Streamlit

```bash
streamlit run src/app.py
```

This app will show the information of simulations, interpolation using 24 months of data and 3 months to test.

### Methodology Notes

The current simulation and forecasting design follows the same general direction as the literature you referenced: hybrid demand forecasting with exogenous drivers and Monte Carlo scenario simulation. In practice, that means the pipeline now distinguishes between:

- baseline seasonality
- promotion-driven uplift
- holiday-driven peaks
- macroeconomic demand compression
- microeconomic store disruptions
- operational zero-sales vs genuine low-demand zeros

For a detailed explanation of the behavioral assumptions and output files, see [docs/event_simulation_methodology.md](docs/event_simulation_methodology.md).

---