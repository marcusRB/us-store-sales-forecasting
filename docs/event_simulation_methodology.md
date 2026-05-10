# Event-Driven Simulation Methodology

## Scope

This project now simulates retail demand with explicit causal drivers instead of purely random sales draws. The objective is to make forecasting, anomaly detection, and interpolation behave more like a real retail system where spikes and collapses are often explained by external conditions rather than by noise alone.

The implementation is conceptually aligned with the literature direction in the two references provided by the user: hybrid demand forecasting with exogenous drivers and Monte Carlo scenario simulation for inventory and uncertainty analysis. The source websites blocked automated retrieval during implementation, so this document reflects a literature-consistent operationalization rather than a direct textual reproduction of those papers.

## Core Behavioral Assumptions

### 1. Base demand is persistent, not arbitrary

Each active store-item pair starts from a latent average daily demand level. That baseline is driven by:

- store traffic and store cluster
- item popularity and item demand pattern
- weekly seasonality
- yearly seasonality

This reflects the idea that demand should have a stable structural component before exogenous shocks are applied.

### 2. Demand reacts multiplicatively to observed context

The simulator adjusts base demand with multiplicative factors. This makes event interactions more realistic than additive rules because a promotion during a holiday or a storm during a recession changes the intensity of the same baseline differently.

The adjusted demand is shaped by:

- holiday uplift
- promotion uplift and price elasticity
- macroeconomic stress or rebound
- local store disruptions
- weather penalties
- inflation pressure by category
- regional unemployment pressure

### 3. Zero sales are not all the same

Zero sales are generated through several mechanisms rather than one generic zero process.

- stockout: demand exists but inventory is effectively unavailable
- inventory system failure: operational zero caused by a hard store-level disruption
- macro demand collapse: discretionary demand temporarily vanishes under macro stress
- weather disruption: storms reduce traffic enough to produce observed zeros
- intermittent zero sale: low-frequency/noisy item behavior

This matters because interpolation and forecasting should treat a stockout very differently from a true lack of demand.

## Event Layers

### Holidays

The holiday calendar contains public and commercial events such as Thanksgiving, Black Friday, Christmas, Labor Day, Memorial Day, and Super Bowl. Each holiday has an impact factor, and each store cluster has a holiday uplift coefficient.

Interpretation:

- commercial holidays produce stronger peak demand for relevant items
- public holidays change foot traffic and basket composition
- holiday effects are store-sensitive, not uniform

### Promotions

Promotions are generated every month at store-item level. Each promotion has:

- discount type
- discount value
- duration window
- uplift multiplier derived from price elasticity and item effectiveness

Interpretation:

- higher elasticity increases demand response
- promotions should create explainable peaks rather than unexplained outliers

### Macroeconomic events

Macro events are generated from monthly oil, inflation, and interest-rate stress indicators. High combined stress produces a `macro_stress` window, while low stress can create a `consumer_rebound` window.

Interpretation:

- macro stress suppresses demand and increases zero-sale risk, especially for discretionary products
- rebounds lift demand, but with lower zero-sale risk

### Microeconomic events

Store-specific disruptions are generated independently for each store. Examples include:

- renovation
- road closure
- local strike
- power outage
- inventory system failure

Interpretation:

- these events reduce demand locally
- some events also directly increase the probability of observed zero sales
- severe operational events can force temporary zeros even when underlying demand is positive

### Weather

Daily weather is simulated by weather zone using sinusoidal seasonality with autocorrelated noise. Weather contributes traffic penalties through:

- extreme heat
- extreme cold
- storms

Interpretation:

- weather does not create arbitrary randomness; it modifies store accessibility and shopping intensity

## Demand Pattern Logic

Items are generated with one of three demand types.

### Regular

Used for fast-moving products. Daily realized sales are sampled with a Poisson-like process around adjusted demand.

### Sporadic

Used for products with many low-activity days. These combine positive demand with additional zero inflation.

### Lumpy

Used for highly variable products with occasional bursts. These are generated with a negative binomial process to preserve overdispersion.

## Anomaly Attribution

The anomaly analysis does not only ask whether a day is unusual. It also asks why.

Daily anomalies are classified using rolling z-scores and event-aware rules.

### Anomaly types

- holiday_peak
- promotion_peak
- macro_stress_drop
- micro_disruption
- zero_sale_cluster
- peak_surge
- statistical_outlier

### Detection idea

- first compute the daily aggregate series
- then compare each day against a rolling 14-day local baseline
- then override or refine the label using observed event counts and zero-sale concentration

This is more useful for forecasting diagnostics because it separates explainable anomalies from unexplained residuals.

## Forecasting and Interpolation Design

### Train/test horizon

- train window: 2024-01-01 to 2026-01-31
- test window: 2026-02-01 to 2026-04-30

This gives a 24-month training history and a 3-month holdout period as requested.

### Forecast model

The current pipeline uses `HistGradientBoostingRegressor` with event-aware features:

- temporal features
- store features
- item features
- holiday/promotion flags
- macro and micro event indicators
- weather and regional economic variables

### Interpolation logic

Artificial missingness is injected into the test window with higher probability around complex event periods. This makes interpolation harder and more realistic than uniformly random missing rows.

### Monte Carlo layer

The Monte Carlo simulation does two things:

- perturbs the model prediction with event-conditioned uncertainty
- applies a stochastic zero-event collapse probability

The uncertainty increases under:

- holidays
- promotions
- macro stress
- micro disruptions
- peak days
- elevated inflation

The scenario output provides:

- mean prediction
- median prediction
- 5th percentile
- 95th percentile

This turns the forecasting output into a distribution rather than a single point estimate.

## Output Files

### Generated tables

- `src/data/*.parquet`
- `src/data/us_store_sales.db`

### Time-series and anomaly outputs

- `output/analysis/daily_event_summary.csv`
- `output/analysis/daily_anomalies.csv`
- `output/analysis/anomaly_cause_summary.csv`
- `output/analysis/anomaly_report.md`
- `output/plots/time_series/ts_decomposition.png`
- `output/plots/time_series/anomalies_by_cause.png`

### Forecasting outputs

- `output/forecast_test_rows.csv`
- `output/interpolation_masked_rows.csv`
- `output/forecast_daily_aggregate.csv`
- `output/forecasting_report.md`
- `output/ml_interpolation_metrics.txt`
- `output/plots/ml_forecasting/daily_forecast_test_horizon.png`
- `output/plots/ml_forecasting/interpolation_masked_test_rows.png`

## Recommended Execution Order

Run everything from the project root and use the `py311-sales` conda environment.

```bash
conda run -n py311-sales python src/generate_full_dataset.py
conda run -n py311-sales python src/time_series_anomalies.py
conda run -n py311-sales python src/ml_forecasting.py
```

## Why This Helps Forecasting

Retail forecasting degrades when structurally different events are collapsed into one undifferentiated noise term. The current simulator avoids that by separating:

- deterministic seasonality
- price and promotion response
- macro demand compression
- local operational disruptions
- weather shocks
- true zero demand vs operational zero sales

That separation is what makes downstream clustering, forecasting, anomaly analysis, and interpolation more defensible.