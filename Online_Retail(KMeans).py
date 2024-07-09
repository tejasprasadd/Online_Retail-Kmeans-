import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from ucimlrepo import fetch_ucirepo
import numpy as np

# Fetching the dataset
online_retail = fetch_ucirepo(id=352)
data = online_retail.data

# Converting features and original data to DataFrame
features_df = pd.DataFrame(data.features, columns=data.headers)

#Using 'Quantity' and 'UnitPrice' for simplicity
X = features_df[['Quantity', 'UnitPrice']]

# Filtering non-positive values
X = X[(X['Quantity'] > 0) & (X['UnitPrice'] > 0)]

# Apply log transformation for better visualization
X_log = np.log1p(X)

# Just to check the datatype of X
print(type(X_log))  # <class 'pandas.core.frame.DataFrame'>

# Create a figure for plotting
plt.figure(figsize=(12, 6))

# Before Clustering
plt.subplot(1, 2, 1)
plt.scatter(X_log['Quantity'], X_log['UnitPrice'], c='gray', marker='o')
plt.xlabel('Log Quantity')
plt.ylabel('Log Unit Price')
plt.title('Before Clustering')

# K-means clustering
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(X_log)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# After Clustering
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_log['Quantity'], X_log['UnitPrice'], c=labels, cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
plt.xlabel('Log Quantity')
plt.ylabel('Log Unit Price')
plt.title('After Clustering')
handles = scatter.legend_elements()[0]
plt.legend(handles=handles, labels=['Cluster 1', 'Cluster 2', 'Cluster 3','Cluster 4','Cluster 5', 'Centroids'], loc='upper right')

plt.tight_layout()
plt.show()

# Selecting only numeric columns for the correlation matrix
numeric_features_df = features_df.select_dtypes(include=[np.number])

# Computing and printing the correlation matrix
corr_matrix = numeric_features_df.corr()
print("Correlation matrix:\n", corr_matrix)

# Ploting the correlation matrix
plt.figure(figsize=(10, 8))
plt.matshow(corr_matrix, fignum=1)
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.colorbar()
plt.title('Correlation Matrix', pad=20)
plt.show()
