import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots', datetime.now().strftime('%Y%m%d_%H%M%S'))

os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data():
    sales = pl.read_parquet(os.path.join(DATA_DIR, 'sales.parquet'))
    stores = pl.read_parquet(os.path.join(DATA_DIR, 'stores.parquet'))
    items = pl.read_parquet(os.path.join(DATA_DIR, 'items.parquet'))
    return sales, stores, items

def run_eda():
    sales, stores, items = load_data()
    print("Running EDA...")
    
    # Total sales over time
    sales_pdf = sales.group_by('date').agg(pl.col('units_sold').sum()).sort('date').to_pandas()
    plt.figure(figsize=(12, 6))
    plt.plot(sales_pdf['date'], sales_pdf['units_sold'])
    plt.title('Total Units Sold Over Time')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    plt.savefig(os.path.join(PLOTS_DIR, 'total_sales_over_time.png'))
    plt.close()

    # Sales by store type
    sales_stores = sales.join(stores, on='store_id', how='left')
    sales_store_type = sales_stores.group_by('store_type').agg(pl.col('units_sold').sum()).to_pandas()
    plt.figure(figsize=(8, 5))
    sns.barplot(x='store_type', y='units_sold', data=sales_store_type)
    plt.title('Total Sales by Store Type')
    plt.savefig(os.path.join(PLOTS_DIR, 'sales_by_store_type.png'))
    plt.close()

    with open(os.path.join(OUTPUT_DIR, 'statistical_info.txt'), 'w') as f:
        f.write("Dataset Statistics\n")
        f.write(f"Total Sales Records: {sales.height}\n")
        f.write(f"Total Units Sold: {sales['units_sold'].sum()}\n")
        f.write(f"Average Units Sold per record: {sales['units_sold'].mean():.2f}\n")
        
    print(f"EDA complete. Plots saved to {PLOTS_DIR}")

if __name__ == '__main__':
    run_eda()
