import os
import sqlite3
import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2026, 4, 30)
N_STORES = 100
N_ITEMS = 1000
SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SQLITE_PATH = os.path.join(DATA_DIR, "us_store_sales.db")

np.random.seed(SEED)
RNG = np.random.default_rng(SEED)
os.makedirs(DATA_DIR, exist_ok=True)

STATE_WEIGHTS = {
    "CA": 0.16,
    "TX": 0.13,
    "NY": 0.10,
    "FL": 0.11,
    "IL": 0.09,
    "PA": 0.08,
    "OH": 0.08,
    "GA": 0.08,
    "NC": 0.09,
    "MI": 0.08,
}
STATE_REGION = {
    "CA": "West",
    "TX": "South",
    "NY": "Northeast",
    "FL": "South",
    "IL": "Midwest",
    "PA": "Northeast",
    "OH": "Midwest",
    "GA": "South",
    "NC": "South",
    "MI": "Midwest",
}
CITY_BY_STATE = {
    "CA": ["Los Angeles", "San Diego", "Sacramento", "San Jose"],
    "TX": ["Houston", "Dallas", "Austin", "San Antonio"],
    "NY": ["New York", "Buffalo", "Albany", "Rochester"],
    "FL": ["Miami", "Orlando", "Tampa", "Jacksonville"],
    "IL": ["Chicago", "Aurora", "Naperville", "Rockford"],
    "PA": ["Philadelphia", "Pittsburgh", "Allentown", "Erie"],
    "OH": ["Columbus", "Cleveland", "Cincinnati", "Toledo"],
    "GA": ["Atlanta", "Savannah", "Augusta", "Athens"],
    "NC": ["Charlotte", "Raleigh", "Durham", "Greensboro"],
    "MI": ["Detroit", "Grand Rapids", "Lansing", "Ann Arbor"],
}
COUNTY_BY_STATE = {
    "CA": ["Los Angeles", "San Diego", "Orange", "Santa Clara"],
    "TX": ["Harris", "Dallas", "Travis", "Bexar"],
    "NY": ["Kings", "Queens", "New York", "Erie"],
    "FL": ["Miami-Dade", "Orange", "Hillsborough", "Duval"],
    "IL": ["Cook", "DuPage", "Lake", "Will"],
    "PA": ["Philadelphia", "Allegheny", "Montgomery", "Berks"],
    "OH": ["Franklin", "Cuyahoga", "Hamilton", "Lucas"],
    "GA": ["Fulton", "Chatham", "Richmond", "Clarke"],
    "NC": ["Mecklenburg", "Wake", "Durham", "Guilford"],
    "MI": ["Wayne", "Kent", "Ingham", "Washtenaw"],
}
PRICE_SENSITIVITY_BY_CLUSTER = [0.85, 0.95, 1.0, 1.08, 1.16]
HOLIDAY_UPLIFT_BY_CLUSTER = [1.05, 1.12, 1.2, 1.28, 1.15]
TRAFFIC_FACTOR_BY_CLUSTER = [0.88, 0.96, 1.0, 1.08, 1.18]
WEATHER_ZONE_BY_REGION = {"West": 0, "South": 1, "Midwest": 2, "Northeast": 3}
SEASONALITY_MONTHS = {
    "none": [1.0] * 12,
    "summer_peak": [0.96, 0.95, 0.97, 0.99, 1.04, 1.12, 1.18, 1.16, 1.06, 1.0, 0.98, 0.99],
    "winter_peak": [1.14, 1.08, 1.02, 0.98, 0.96, 0.95, 0.94, 0.95, 0.98, 1.01, 1.08, 1.18],
    "december_peak": [0.93, 0.93, 0.95, 0.98, 1.0, 1.02, 1.03, 1.01, 0.99, 1.02, 1.1, 1.34],
}
WEEKLY_PATTERNS = {
    "weekday": [0.98, 1.0, 1.0, 1.02, 1.06, 1.08, 0.92],
    "weekend": [0.9, 0.92, 0.95, 1.0, 1.08, 1.24, 1.18],
    "steady": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}


def save_table(df, name, sqlite_conn):
    parquet_path = os.path.join(DATA_DIR, f"{name}.parquet")
    df.write_parquet(parquet_path, compression="zstd")
    df.to_pandas().to_sql(name, sqlite_conn, if_exists="replace", index=False)


def nth_weekday(year, month, weekday, n):
    current = date(year, month, 1)
    while current.weekday() != weekday:
        current += timedelta(days=1)
    current += timedelta(days=(n - 1) * 7)
    return current


def last_weekday(year, month, weekday):
    if month == 12:
        current = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        current = date(year, month + 1, 1) - timedelta(days=1)
    while current.weekday() != weekday:
        current -= timedelta(days=1)
    return current


def clamp(value, low, high):
    return max(low, min(high, value))


def generate_calendar():
    dates = pl.date_range(START_DATE, END_DATE, interval="1d", eager=True)
    return pl.DataFrame({"date": dates}).with_columns([
        pl.col("date").dt.weekday().alias("day_of_week"),
        pl.col("date").dt.week().alias("week"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.quarter().alias("quarter"),
        pl.col("date").dt.year().alias("year"),
        (pl.col("date").dt.weekday() >= 5).alias("is_weekend"),
    ])


def generate_stores():
    states = np.array(list(STATE_WEIGHTS.keys()))
    state_probs = np.array(list(STATE_WEIGHTS.values()))
    chosen_states = RNG.choice(states, size=N_STORES, p=state_probs)
    cluster_id = RNG.choice(np.arange(5), size=N_STORES, p=[0.25, 0.2, 0.2, 0.18, 0.17])
    store_types = RNG.choice(["Urban", "Suburban", "Rural"], size=N_STORES, p=[0.4, 0.42, 0.18])
    rows = []
    for index in range(N_STORES):
        state = chosen_states[index]
        region = STATE_REGION[state]
        cluster = int(cluster_id[index])
        store_type = str(store_types[index])
        type_scale = {"Urban": 1.18, "Suburban": 1.0, "Rural": 0.76}[store_type]
        traffic = int(np.round(clamp(RNG.normal(1250 * type_scale * TRAFFIC_FACTOR_BY_CLUSTER[cluster], 180), 350, 2800)))
        sqft = int(np.round(clamp(RNG.normal(36000 * type_scale, 6500), 9000, 82000)))
        rows.append({
            "store_id": index + 1,
            "store_name": f"Store_{index + 1:03d}",
            "city": RNG.choice(CITY_BY_STATE[state]),
            "county": RNG.choice(COUNTY_BY_STATE[state]),
            "state": state,
            "region": region,
            "store_type": store_type,
            "size_sqft": sqft,
            "avg_customer_traffic": traffic,
            "cluster_id": cluster,
            "price_sensitivity": round(PRICE_SENSITIVITY_BY_CLUSTER[cluster], 2),
            "holiday_uplift": round(HOLIDAY_UPLIFT_BY_CLUSTER[cluster], 2),
            "traffic_band": "high" if traffic > 1500 else "medium" if traffic > 1000 else "low",
            "unemployment_region_id": ["West", "South", "Midwest", "Northeast"].index(region),
            "weather_zone_id": WEATHER_ZONE_BY_REGION[region],
        })
    return pl.DataFrame(rows)


def generate_items():
    categories = (["Grocery"] * 400) + (["Health & Beauty"] * 300) + (["Household Essentials"] * 300)
    RNG.shuffle(categories)
    category_subcats = {
        "Grocery": ["Fresh Food", "Other Food", "Frozen", "Beverages"],
        "Health & Beauty": ["Personal Care", "Cosmetics", "Pharmacy", "Wellness"],
        "Household Essentials": ["Cleaning", "Paper", "Laundry", "Kitchen"],
    }
    rows = []
    for item_id in range(1, N_ITEMS + 1):
        category = categories[item_id - 1]
        if category == "Grocery":
            base_price = clamp(RNG.lognormal(mean=np.log(4.5), sigma=0.55), 0.8, 22)
            weight = clamp(RNG.normal(1.1, 0.7), 0.1, 4.0)
            is_perishable = RNG.random() < 0.62
            pattern_probs = [0.72, 0.22, 0.06]
        elif category == "Health & Beauty":
            base_price = clamp(RNG.lognormal(mean=np.log(8.5), sigma=0.5), 2.5, 35)
            weight = clamp(RNG.normal(0.45, 0.18), 0.05, 1.4)
            is_perishable = RNG.random() < 0.08
            pattern_probs = [0.5, 0.35, 0.15]
        else:
            base_price = clamp(RNG.lognormal(mean=np.log(10.5), sigma=0.45), 3.0, 42)
            weight = clamp(RNG.normal(1.6, 0.9), 0.15, 5.5)
            is_perishable = RNG.random() < 0.12
            pattern_probs = [0.55, 0.28, 0.17]
        rows.append({
            "item_id": item_id,
            "sku_code": f"SKU-{item_id:04d}",
            "description": f"{category} Product {item_id}",
            "category": category,
            "subcategory": RNG.choice(category_subcats[category]),
            "base_price": round(base_price, 2),
            "weight_kg_per_unit": round(weight, 3),
            "weight_std_kg": round(max(0.03, weight * 0.12), 3),
            "is_weight_variable": bool(RNG.random() < (0.3 if category == "Grocery" else 0.08)),
            "is_perishable": bool(is_perishable),
            "demand_pattern": RNG.choice(["Regular", "Sporadic", "Lumpy"], p=pattern_probs),
            "cluster_id": int(RNG.integers(0, 10)),
            "seasonality_profile": RNG.choice(["none", "summer_peak", "winter_peak", "december_peak"], p=[0.3, 0.22, 0.2, 0.28]),
            "weekly_profile": RNG.choice(["weekday", "weekend", "steady"], p=[0.35, 0.4, 0.25]),
            "base_effectiveness": round(clamp(RNG.normal(1.0, 0.15), 0.65, 1.45), 3),
            "elasticity": round(clamp(RNG.normal(-1.18, 0.35), -2.2, -0.35), 2),
            "zero_inflation_prob": round(clamp(RNG.beta(1.8, 8.0), 0.0, 0.45), 3),
            "overdispersion": round(clamp(RNG.gamma(shape=2.0, scale=1.1), 0.6, 6.0), 3),
            "avg_daily_demand": round(clamp(RNG.lognormal(mean=np.log(1.8), sigma=0.8), 0.05, 12), 3),
            "popularity_score": round(clamp(RNG.beta(2.2, 4.8), 0.02, 0.95), 3),
        })
    return pl.DataFrame(rows)


def random_walk(length, start, drift, scale, floor):
    values = [start]
    for _ in range(length - 1):
        values.append(max(floor, values[-1] + drift + RNG.normal(0, scale)))
    return np.round(values, 3)


def generate_economic():
    months = pl.date_range(START_DATE.replace(day=1), END_DATE.replace(day=1), interval="1mo", eager=True)
    month_list = months.to_list()
    length = len(month_list)
    oil_base = random_walk(length, 74.0, 0.05, 1.6, 48.0)
    inflation_base = random_walk(length, 2.7, 0.01, 0.08, 1.6)
    rate_base = random_walk(length, 3.7, 0.0, 0.04, 2.0)
    oil = pl.DataFrame({
        "date": months,
        "price_usd_per_barrel": oil_base,
        "oil_stress_index": np.round((oil_base - np.mean(oil_base)) / np.std(oil_base), 3),
    })
    inflation = pl.DataFrame({
        "date": months,
        "cpi_yoy_pct": inflation_base,
        "inflation_stress_index": np.round((inflation_base - np.mean(inflation_base)) / np.std(inflation_base), 3),
    })
    unemployment_rows = []
    for region_id, start_value in {0: 4.4, 1: 4.9, 2: 4.7, 3: 4.3}.items():
        series = random_walk(length, start_value, 0.0, 0.07, 2.8)
        seasonal = np.array([0.18 * np.sin((index / 12.0) * 2 * np.pi + region_id) for index in range(length)])
        adjusted = np.round(np.clip(series + seasonal, 2.8, 8.5), 3)
        for month, rate in zip(month_list, adjusted):
            unemployment_rows.append({"date": month, "region_id": region_id, "rate_pct": rate})
    unemployment = pl.DataFrame(unemployment_rows)
    interest = pl.DataFrame({
        "date": months,
        "fed_funds_rate_pct": rate_base,
        "credit_stress_index": np.round((rate_base - np.mean(rate_base)) / np.std(rate_base), 3),
    })
    return oil, inflation, unemployment, interest


def generate_holidays():
    rows = []
    holiday_id = 1
    for year in range(START_DATE.year, END_DATE.year + 1):
        thanksgiving = nth_weekday(year, 11, 3, 4)
        black_friday = thanksgiving + timedelta(days=1)
        labor_day = nth_weekday(year, 9, 0, 1)
        memorial_day = last_weekday(year, 5, 0)
        mlk_day = nth_weekday(year, 1, 0, 3)
        super_bowl = nth_weekday(year, 2, 6, 2)
        holiday_defs = [
            (date(year, 1, 1), "New Year", 1.18, "Public"),
            (mlk_day, "MLK Day", 1.04, "Public"),
            (super_bowl, "Super Bowl", 1.22, "Commercial"),
            (memorial_day, "Memorial Day", 1.12, "Public"),
            (date(year, 7, 4), "Independence Day", 1.28, "Public"),
            (labor_day, "Labor Day", 1.1, "Public"),
            (thanksgiving, "Thanksgiving", 1.52, "Public"),
            (black_friday, "Black Friday", 1.85, "Commercial"),
            (date(year, 12, 24), "Christmas Eve", 1.34, "Commercial"),
            (date(year, 12, 25), "Christmas", 1.68, "Public"),
        ]
        for holiday_date, name, impact, holiday_type in holiday_defs:
            if START_DATE.date() <= holiday_date <= END_DATE.date():
                rows.append({
                    "holiday_id": holiday_id,
                    "date": holiday_date,
                    "holiday_name": name,
                    "is_national": True,
                    "affected_states": None,
                    "impact_factor": round(impact, 2),
                    "type": holiday_type,
                })
                holiday_id += 1
    return pl.DataFrame(rows).sort("date")


def attach_holidays_to_calendar(calendar, holidays):
    return calendar.join(
        holidays.select(["holiday_id", "date", "holiday_name", "impact_factor"]),
        on="date",
        how="left",
    ).with_columns([
        pl.col("holiday_id").is_not_null().alias("is_holiday"),
        pl.col("impact_factor").fill_null(1.0).alias("holiday_impact_factor"),
        pl.col("holiday_name").fill_null("none"),
    ])


def generate_macro_events(oil, inflation, interest):
    monthly = oil.to_pandas().merge(inflation.to_pandas(), on="date").merge(interest.to_pandas(), on="date")
    monthly["stress_score"] = monthly["oil_stress_index"] * 0.25 + monthly["inflation_stress_index"] * 0.45 + monthly["credit_stress_index"] * 0.3
    rows = []
    event_id = 1
    for _, row in monthly.sort_values("stress_score", ascending=False).head(3).iterrows():
        start_date = pd.Timestamp(row["date"]).date()
        stress_score = float(row["stress_score"])
        rows.append({
            "event_id": event_id,
            "start_date": start_date,
            "end_date": start_date + timedelta(days=27),
            "event_type": "macro_stress",
            "description": "High inflation and credit tightening reduce discretionary demand.",
            "severity": round(clamp(0.5 + stress_score * 0.15, 0.4, 0.95), 3),
            "demand_multiplier": round(clamp(0.92 - stress_score * 0.08, 0.65, 0.95), 3),
            "zero_sale_risk_add": round(clamp(0.01 + stress_score * 0.015, 0.0, 0.07), 3),
            "trigger_source": "inflation_oil_rates",
        })
        event_id += 1
    best_month = monthly.sort_values("stress_score", ascending=True).iloc[0]
    best_start = pd.Timestamp(best_month["date"]).date()
    rows.append({
        "event_id": event_id,
        "start_date": best_start,
        "end_date": best_start + timedelta(days=20),
        "event_type": "consumer_rebound",
        "description": "Cooling inflation and lower rate pressure lift discretionary demand.",
        "severity": 0.45,
        "demand_multiplier": 1.08,
        "zero_sale_risk_add": 0.0,
        "trigger_source": "macro_relief",
    })
    return pl.DataFrame(rows).sort("start_date")


def generate_micro_events(df_stores):
    micro_types = {
        "renovation": {"multiplier": 0.55, "zero_risk": 0.04},
        "road_closure": {"multiplier": 0.72, "zero_risk": 0.02},
        "local_strike": {"multiplier": 0.38, "zero_risk": 0.08},
        "power_outage": {"multiplier": 0.1, "zero_risk": 0.24},
        "inventory_system_failure": {"multiplier": 0.0, "zero_risk": 0.65},
    }
    rows = []
    event_id = 1
    date_span = (END_DATE - START_DATE).days - 20
    for store in df_stores.iter_rows(named=True):
        for _ in range(int(RNG.integers(2, 5))):
            event_type = RNG.choice(list(micro_types.keys()), p=[0.32, 0.25, 0.18, 0.15, 0.1])
            start = START_DATE.date() + timedelta(days=int(RNG.integers(0, max(1, date_span))))
            duration = int(RNG.integers(3, 11 if event_type != "renovation" else 18))
            meta = micro_types[event_type]
            rows.append({
                "event_id": event_id,
                "store_id": store["store_id"],
                "start_date": start,
                "end_date": start + timedelta(days=duration),
                "type": event_type,
                "demand_multiplier": meta["multiplier"],
                "zero_sale_risk_add": meta["zero_risk"],
                "severity": round(clamp(RNG.normal(0.55, 0.18), 0.2, 0.95), 3),
            })
            event_id += 1
    return pl.DataFrame(rows).sort(["store_id", "start_date"])


def generate_promotions(df_stores, df_items):
    item_pdf = df_items.select(["item_id", "elasticity", "base_effectiveness", "popularity_score"]).to_pandas()
    item_weights = item_pdf["popularity_score"].to_numpy()
    item_weights = item_weights / item_weights.sum()
    promo_rows = []
    promo_id = 1
    months = pd.date_range(START_DATE.replace(day=1), END_DATE.replace(day=1), freq="MS")
    for store in df_stores.iter_rows(named=True):
        for month_start in months:
            for _ in range(int(RNG.integers(1, 4))):
                sampled_idx = int(RNG.choice(item_pdf.index, p=item_weights))
                item = item_pdf.iloc[sampled_idx]
                discount_value = round(clamp(RNG.uniform(0.1, 0.38), 0.08, 0.45), 3)
                start_offset = int(RNG.integers(0, 20))
                duration = int(RNG.integers(3, 10))
                start_date = pd.Timestamp(month_start).date() + timedelta(days=start_offset)
                end_date = min(start_date + timedelta(days=duration), END_DATE.date())
                promo_multiplier = clamp(1 + discount_value * abs(item["elasticity"]) * item["base_effectiveness"] * store["price_sensitivity"], 1.03, 1.9)
                promo_rows.append({
                    "promotion_id": promo_id,
                    "store_id": store["store_id"],
                    "item_id": int(item["item_id"]),
                    "start_date": start_date,
                    "end_date": end_date,
                    "discount_type": RNG.choice(["percentage_off", "bogo", "bulk_discount", "weight_step"], p=[0.55, 0.15, 0.2, 0.1]),
                    "discount_value": discount_value,
                    "promo_multiplier": round(promo_multiplier, 3),
                })
                promo_id += 1
    return pl.DataFrame(promo_rows).sort(["store_id", "start_date"])


def generate_weather(df_stores, calendar):
    zone_base = {
        0: {"temp": 18.5, "amplitude": 8.0, "precip_scale": 1.8},
        1: {"temp": 22.0, "amplitude": 7.0, "precip_scale": 2.7},
        2: {"temp": 10.5, "amplitude": 14.0, "precip_scale": 2.1},
        3: {"temp": 11.0, "amplitude": 12.5, "precip_scale": 2.0},
    }
    rows = []
    for store in df_stores.iter_rows(named=True):
        zone = store["weather_zone_id"]
        params = zone_base[zone]
        noise = 0.0
        for current_date in calendar["date"].to_list():
            day_of_year = current_date.timetuple().tm_yday
            noise = 0.72 * noise + RNG.normal(0, 1.5)
            temp = params["temp"] + params["amplitude"] * np.sin((2 * np.pi * day_of_year / 365.25) - 1.4) + noise
            precip = max(0.0, RNG.gamma(shape=1.8, scale=params["precip_scale"]))
            snowfall = max(0.0, precip * 0.35) if temp < 1.0 and zone in {2, 3} else 0.0
            rows.append({
                "store_id": store["store_id"],
                "date": current_date,
                "temp_celsius": round(temp, 2),
                "precipitation_mm": round(precip, 2),
                "snowfall_mm": round(snowfall, 2),
                "is_extreme_heat": temp >= 35,
                "is_extreme_cold": temp <= -6,
                "is_storm": precip >= 12,
            })
    return pl.DataFrame(rows)


def build_macro_daily_lookup(df_macro, dates):
    lookup = {current_date: {"macro_multiplier": 1.0, "macro_zero_risk_add": 0.0, "macro_event_id": None, "macro_event_type": None} for current_date in dates}
    for row in df_macro.iter_rows(named=True):
        current = row["start_date"]
        while current <= row["end_date"]:
            lookup[current]["macro_multiplier"] *= row["demand_multiplier"]
            lookup[current]["macro_zero_risk_add"] += row["zero_sale_risk_add"]
            lookup[current]["macro_event_id"] = row["event_id"]
            lookup[current]["macro_event_type"] = row["event_type"]
            current += timedelta(days=1)
    return lookup


def build_micro_daily_lookup(df_micro):
    lookup = {}
    for row in df_micro.iter_rows(named=True):
        current = row["start_date"]
        while current <= row["end_date"]:
            lookup[(row["store_id"], current)] = {
                "micro_multiplier": row["demand_multiplier"],
                "micro_zero_risk_add": row["zero_sale_risk_add"],
                "micro_event_id": row["event_id"],
                "micro_event_type": row["type"],
            }
            current += timedelta(days=1)
    return lookup


def build_promo_lookup(df_promotions):
    lookup = {}
    for row in df_promotions.iter_rows(named=True):
        current = row["start_date"]
        while current <= row["end_date"]:
            lookup[(row["store_id"], row["item_id"], current)] = {
                "promotion_id": row["promotion_id"],
                "discount_type": row["discount_type"],
                "discount_value": row["discount_value"],
                "promo_multiplier": row["promo_multiplier"],
            }
            current += timedelta(days=1)
    return lookup


def build_weather_lookup(df_weather):
    return {(row["store_id"], row["date"]): row for row in df_weather.iter_rows(named=True)}


def build_holiday_lookup(df_holidays):
    return {row["date"]: row for row in df_holidays.iter_rows(named=True)}


def build_unemployment_lookup(df_unemployment):
    return {(row["date"].strftime("%Y-%m"), row["region_id"]): row["rate_pct"] for row in df_unemployment.iter_rows(named=True)}


def generate_sales(df_stores, df_items, calendar, holidays, promotions, weather, inflation, unemployment, macro_events, micro_events):
    stores = df_stores.to_pandas()
    items = df_items.to_pandas()
    dates = calendar["date"].to_list()
    holiday_lookup = build_holiday_lookup(holidays)
    promo_lookup = build_promo_lookup(promotions)
    weather_lookup = build_weather_lookup(weather)
    macro_lookup = build_macro_daily_lookup(macro_events, dates)
    micro_lookup = build_micro_daily_lookup(micro_events)
    inflation_lookup = {row["date"].strftime("%Y-%m"): row["cpi_yoy_pct"] for row in inflation.iter_rows(named=True)}
    unemployment_lookup = build_unemployment_lookup(unemployment)
    item_weight = items["popularity_score"].to_numpy()
    item_weight = item_weight / item_weight.sum()
    sales_rows = []
    for _, store in stores.iterrows():
        assortment_size = int(clamp(16 + store["cluster_id"] * 3 + store["avg_customer_traffic"] / 320, 18, 36))
        active_item_ids = RNG.choice(items["item_id"].to_numpy(), size=assortment_size, replace=False, p=item_weight)
        assortment = items[items["item_id"].isin(active_item_ids)].copy()
        assortment["pair_base_demand"] = (assortment["avg_daily_demand"] * (store["avg_customer_traffic"] / 1150.0) * TRAFFIC_FACTOR_BY_CLUSTER[int(store["cluster_id"])] * RNG.lognormal(mean=-0.05, sigma=0.32, size=len(assortment))).clip(0.03, 18.0)
        for item in assortment.itertuples(index=False):
            previous_stockout = False
            previous_units = 0
            for current_date in dates:
                month_key = current_date.strftime("%Y-%m")
                inflation_rate = inflation_lookup[month_key]
                unemployment_rate = unemployment_lookup[(month_key, int(store["unemployment_region_id"]))]
                holiday = holiday_lookup.get(current_date)
                promo = promo_lookup.get((int(store["store_id"]), int(item.item_id), current_date))
                weather_row = weather_lookup[(int(store["store_id"]), current_date)]
                macro_row = macro_lookup[current_date]
                micro_row = micro_lookup.get((int(store["store_id"]), current_date), {
                    "micro_multiplier": 1.0,
                    "micro_zero_risk_add": 0.0,
                    "micro_event_id": None,
                    "micro_event_type": None,
                })
                weekly_factor = WEEKLY_PATTERNS[item.weekly_profile][current_date.weekday()]
                seasonal_factor = SEASONALITY_MONTHS[item.seasonality_profile][current_date.month - 1]
                holiday_factor = (holiday["impact_factor"] * store["holiday_uplift"]) if holiday else 1.0
                promo_multiplier = promo["promo_multiplier"] if promo else 1.0
                discount_value = promo["discount_value"] if promo else 0.0
                discounted_price = round(item.base_price * (1 - discount_value), 2) if promo else float(item.base_price)
                weather_penalty = 1.0
                if weather_row["is_storm"]:
                    weather_penalty *= 0.82
                if weather_row["is_extreme_heat"]:
                    weather_penalty *= 0.92
                if weather_row["is_extreme_cold"]:
                    weather_penalty *= 0.88
                if item.category == "Health & Beauty":
                    inflation_pressure = 1.0 - clamp((inflation_rate - 2.2) * 0.018, 0.0, 0.18)
                elif item.category == "Household Essentials":
                    inflation_pressure = 1.0 - clamp((inflation_rate - 2.2) * 0.01, 0.0, 0.1)
                else:
                    inflation_pressure = 1.0 - clamp((inflation_rate - 2.2) * 0.006, 0.0, 0.06)
                unemployment_penalty = 1.0 - clamp((unemployment_rate - 4.2) * 0.015, 0.0, 0.11)
                price_factor = (item.base_price / max(discounted_price, 0.25)) ** abs(item.elasticity)
                base_demand = item.pair_base_demand * weekly_factor * seasonal_factor
                true_demand = base_demand * holiday_factor * promo_multiplier * weather_penalty
                true_demand *= macro_row["macro_multiplier"] * micro_row["micro_multiplier"] * inflation_pressure * unemployment_penalty * price_factor
                true_demand = max(0.0, true_demand)
                stockout_risk = 0.012 + (0.018 if item.is_perishable else 0.004)
                if previous_units > max(2.0, item.pair_base_demand * 1.7):
                    stockout_risk += 0.014
                if previous_stockout:
                    stockout_risk += 0.01
                zero_sale_risk = item.zero_inflation_prob * 0.35 + macro_row["macro_zero_risk_add"] + micro_row["micro_zero_risk_add"]
                if weather_row["is_storm"]:
                    zero_sale_risk += 0.018
                if inflation_rate > 3.6 and item.category == "Health & Beauty":
                    zero_sale_risk += 0.02
                zero_sale_risk = clamp(zero_sale_risk, 0.0, 0.92)
                zero_sale_reason = None
                is_stockout = False
                if micro_row["micro_event_type"] == "inventory_system_failure":
                    units_sold = 0
                    zero_sale_reason = "inventory_system_failure"
                elif RNG.random() < stockout_risk:
                    units_sold = 0
                    is_stockout = True
                    zero_sale_reason = "stockout"
                elif true_demand > 0 and RNG.random() < zero_sale_risk:
                    units_sold = 0
                    if micro_row["micro_event_type"]:
                        zero_sale_reason = micro_row["micro_event_type"]
                    elif macro_row["macro_event_type"] == "macro_stress":
                        zero_sale_reason = "demand_collapse"
                    elif weather_row["is_storm"]:
                        zero_sale_reason = "weather_disruption"
                    else:
                        zero_sale_reason = "intermittent_zero_sale"
                else:
                    if item.demand_pattern == "Regular":
                        units_sold = int(RNG.poisson(max(true_demand, 0.01)))
                    elif item.demand_pattern == "Sporadic":
                        if RNG.random() < item.zero_inflation_prob:
                            units_sold = 0
                        else:
                            units_sold = int(RNG.poisson(max(true_demand * 0.95, 0.01)))
                    else:
                        dispersion = max(1.0, item.overdispersion)
                        probability = dispersion / (dispersion + max(true_demand, 0.01))
                        units_sold = int(RNG.negative_binomial(dispersion, probability))
                if item.is_weight_variable and units_sold > 0:
                    total_weight = max(0.0, RNG.normal(units_sold * item.weight_kg_per_unit, max(0.08, units_sold * item.weight_std_kg)))
                else:
                    total_weight = units_sold * item.weight_kg_per_unit
                promo_flag = promo is not None
                is_holiday = holiday is not None
                event_type = "baseline"
                if zero_sale_reason is not None:
                    event_type = "zero_sale_event"
                elif promo_flag:
                    event_type = "promotion"
                elif is_holiday:
                    event_type = "holiday"
                elif micro_row["micro_event_type"]:
                    event_type = "micro_event"
                elif macro_row["macro_event_type"]:
                    event_type = "macro_event"
                is_peak_day = bool(units_sold > max(6, item.pair_base_demand * 2.0) or (is_holiday and holiday_factor > 1.3) or (promo_flag and promo_multiplier > 1.3))
                if is_peak_day and event_type == "baseline":
                    event_type = "peak_surge"
                should_write = units_sold > 0 or zero_sale_reason is not None or promo_flag or is_holiday or micro_row["micro_event_type"] is not None or macro_row["macro_event_type"] is not None or is_peak_day
                if not should_write:
                    previous_units = units_sold
                    previous_stockout = is_stockout
                    continue
                sales_rows.append({
                    "store_id": int(store["store_id"]),
                    "item_id": int(item.item_id),
                    "date": current_date,
                    "units_sold": int(units_sold),
                    "weight_kg_sold": round(total_weight, 3),
                    "price_per_unit_after_discount": round(discounted_price, 2),
                    "promotion_id": promo["promotion_id"] if promo else None,
                    "is_promo_day": promo_flag,
                    "is_holiday": is_holiday,
                    "holiday_id": holiday["holiday_id"] if holiday else None,
                    "holiday_name": holiday["holiday_name"] if holiday else None,
                    "macro_event_id": macro_row["macro_event_id"],
                    "macro_event_type": macro_row["macro_event_type"],
                    "micro_event_id": micro_row["micro_event_id"],
                    "micro_event_type": micro_row["micro_event_type"],
                    "is_stockout": is_stockout,
                    "zero_sale_reason": zero_sale_reason,
                    "detected_event_type": event_type,
                    "is_peak_day": is_peak_day,
                    "base_demand": round(base_demand, 4),
                    "true_demand": round(true_demand, 4),
                    "inflation_rate": round(float(inflation_rate), 3),
                    "regional_unemployment_rate": round(float(unemployment_rate), 3),
                    "temp_celsius": weather_row["temp_celsius"],
                    "precipitation_mm": weather_row["precipitation_mm"],
                })
                previous_units = units_sold
                previous_stockout = is_stockout
    return pl.from_pandas(pd.DataFrame(sales_rows)).sort(["date", "store_id", "item_id"])


if __name__ == '__main__':
    conn = sqlite3.connect(SQLITE_PATH)
    calendar = generate_calendar()
    stores = generate_stores()
    items = generate_items()
    oil, inflation, unemployment, interest = generate_economic()
    holidays = generate_holidays()
    calendar = attach_holidays_to_calendar(calendar, holidays)
    macro = generate_macro_events(oil, inflation, interest)
    micro = generate_micro_events(stores)
    promotions = generate_promotions(stores, items)
    weather = generate_weather(stores, calendar)
    sales = generate_sales(stores, items, calendar, holidays, promotions, weather, inflation, unemployment, macro, micro)
    save_table(calendar, 'calendar', conn)
    save_table(stores, 'stores', conn)
    save_table(items, 'items', conn)
    save_table(oil, 'oil_price', conn)
    save_table(inflation, 'inflation_rate', conn)
    save_table(unemployment, 'unemployment_rate', conn)
    save_table(interest, 'interest_fed_rate', conn)
    save_table(holidays, 'holidays_events', conn)
    save_table(macro, 'macroeconomic_events', conn)
    save_table(micro, 'microeconomic_events', conn)
    save_table(promotions, 'promotions', conn)
    save_table(weather, 'weather_events', conn)
    save_table(sales, 'sales', conn)
    conn.close()
    print(f'Generated tables in {DATA_DIR}')
    print(f'Sales rows: {sales.height}')
