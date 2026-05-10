import polars as pl
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

np.random.seed(42)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def run_clustering():
    stores = pl.read_parquet(os.path.join(DATA_DIR, 'stores.parquet')).to_pandas()
    items = pl.read_parquet(os.path.join(DATA_DIR, 'items.parquet')).to_pandas()

    # Simple store clustering based on numeric features
    scaler = StandardScaler()
    store_features = stores[['size_sqft', 'avg_customer_traffic']]
    scaled_stores = scaler.fit_transform(store_features)
    kmeans_stores = KMeans(n_clusters=5, random_state=42)
    stores['ml_cluster_id'] = kmeans_stores.fit_predict(scaled_stores)

    # Simple item clustering
    item_features = items[['base_price', 'weight_kg_per_unit', 'elasticity']]
    scaled_items = scaler.fit_transform(item_features)
    kmeans_items = KMeans(n_clusters=5, random_state=42)
    items['ml_cluster_id'] = kmeans_items.fit_predict(scaled_items)
    
    print("Clustering completed. (Models can be saved here)")

if __name__ == '__main__':
    run_clustering()
