# Time Series Anomaly Analysis

This report links daily sales anomalies to explicit simulated retail causes: holidays, promotions, macro stress, micro disruptions, peak surges, and clustered zero-sale events.

## Detection Logic
- Baseline anomaly score: 14-day rolling z-score on aggregated units sold.
- Event-aware overrides: holiday, promotion, macro, micro, stockout, and peak-day indicators from the generator.
- Zero-sale clusters: dates with an unusually high share of rows marked by explicit zero-sale reasons.

## Summary
anomaly_cause,count,avg_units_sold,avg_zero_sale_ratio
holiday_peak,20,9476.85,0.09063371861299323
macro_stress_drop,2,5212.0,0.100438421681945
micro_disruption,731,5983.6675786593705,0.11671446067148879
promotion_peak,31,6703.322580645161,0.1089636358574572
zero_sale_cluster,67,6394.492537313433,0.10701938477483158

