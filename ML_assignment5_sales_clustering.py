# Assignment 5: Sales Data Clustering using K-Means and Hierarchical Clustering
# Dataset: sales_data_sample.csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

df = pd.read_csv("sales_data_sample.csv", encoding="latin1")
print(df.head())

# Choose available numeric sales-related columns.
preferred = ["QUANTITYORDERED", "PRICEEACH", "SALES"]
features = [c for c in preferred if c in df.columns]
if len(features) < 2:
    features = df.select_dtypes(include="number").columns.tolist()
if len(features) < 2:
    raise ValueError("Dataset must contain at least two numeric columns.")

X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method
wcss = []
k_values = range(1, 11)
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.plot(list(k_values), wcss, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# K-Means (change n_clusters after inspecting the elbow graph)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
print("K-Means cluster counts:")
print(pd.Series(kmeans_labels).value_counts().sort_index())

# Hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
hier_labels = hierarchical.fit_predict(X_scaled)
print("Hierarchical cluster counts:")
print(pd.Series(hier_labels).value_counts().sort_index())

# PCA plot for visualizing the clusters in 2D
pca = PCA(n_components=2)
points = pca.fit_transform(X_scaled)
plt.scatter(points[:, 0], points[:, 1], c=kmeans_labels, s=15)
plt.title("K-Means Clusters (PCA view)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
