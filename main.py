
# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import pandas as pd
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt


# ==========================================
# 2. CREATE MODEL FOLDER
# ==========================================

os.makedirs("model", exist_ok=True)


# ==========================================
# 3. LOAD DATASET
# ==========================================

df = pd.read_csv("Mall_Customers.csv")

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())


# ==========================================
# 4. SELECT FEATURES
# ==========================================

# Select Annual Income and Spending Score
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]


# ==========================================
# 5. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeature scaling completed.")


# ==========================================
# 6. TRAIN K-MEANS MODEL
# ==========================================

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nK-Means clustering completed.")


# ==========================================
# 7. EVALUATE MODEL
# ==========================================

score = silhouette_score(X_scaled, df["Cluster"])

print("\nSilhouette Score:", round(score, 3))


# ==========================================
# 8. DISPLAY CUSTOMER CLUSTERS
# ==========================================

print("\nCustomer Cluster Counts:")
print(df["Cluster"].value_counts().sort_index())

print("\nSample customer data with clusters:")
print(df.head(10))


# ==========================================
# 9. SAVE MODEL AND SCALER
# ==========================================

joblib.dump(kmeans, "model/kmeans_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\nModel saved: model/kmeans_model.pkl")
print("Scaler saved: model/scaler.pkl")


# ==========================================
# 10. VISUALIZE CLUSTERS
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis",
    s=60
)

plt.scatter(
    scaler.inverse_transform(kmeans.cluster_centers_)[:, 0],
    scaler.inverse_transform(kmeans.cluster_centers_)[:, 1],
    marker="X",
    s=250,
    label="Cluster Centers"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================
# 11. SAVE CLUSTERED DATASET
# ==========================================

df.to_csv("Mall_Customers.csv", index=False)

print("\nClustered dataset saved: customer_segments.csv")
print("\nProject completed successfully!")
