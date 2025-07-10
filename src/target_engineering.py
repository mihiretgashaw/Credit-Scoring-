import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Load raw data
df = pd.read_csv("data/raw/data.csv", low_memory=False)

# Drop garbage columns
df.drop(columns=["Unnamed: 16", "Unnamed: 17"], inplace=True, errors='ignore')

# Convert TransactionStartTime to datetime
df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

# Snapshot date = 1 day after latest transaction
snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

# RFM CALCULATION
rfm = df.groupby("CustomerId").agg({
    "TransactionStartTime": lambda x: (snapshot_date - x.max()).days,
    "TransactionId": "count",
    "Amount": "sum"
}).reset_index()

rfm.columns = ["CustomerId", "Recency", "Frequency", "Monetary"]

# Scale RFM
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

# Label the high-risk cluster
# The high-risk cluster has: High Recency, Low Frequency, Low Monetary
cluster_profiles = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
high_risk_cluster = cluster_profiles["Frequency"].idxmin()  # or combine all three criteria manually

# Assign high-risk binary label
rfm["is_high_risk"] = (rfm["Cluster"] == high_risk_cluster).astype(int)

# Save just CustomerId and is_high_risk
risk_labels = rfm[["CustomerId", "is_high_risk"]]

# Load processed features and merge
X = np.load("data/processed/processed.npy")
df_full = pd.read_csv("data/raw/data.csv", low_memory=False)
df_full = df_full.merge(risk_labels, on="CustomerId", how="left")

# Fill any missing risk label as 0 (if any customers didn't get clustered)
df_full["is_high_risk"] = df_full["is_high_risk"].fillna(0).astype(int)

# Save target variable
y = df_full["is_high_risk"].values
np.save("data/processed/target.npy", y)
print("✅ Saved proxy target variable to data/processed/target.npy")
