'''
Author: NEFU AB-IN
Date: 2024-11-20 17:28:56
FilePath: \SparkDataPipeline\src\main\python\KMeansClusteringVisualization.py
LastEditTime: 2024-11-20 17:33:59
'''
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Load the clustering results
cluster_results_file = "D:\\Code\\OtherProject\\cs5488_g6\\SparkDataPipeline\\src\\main\\resources\\test_output\\cs.PF_cluster_results.csv"
data = pd.read_csv(cluster_results_file)

# Step 2: Extract coordinates and cluster labels
x1 = data['X1']
x2 = data['X2']
clusters = data['Cluster']

# Step 3: Plot the clustering results
plt.figure(figsize=(10, 6))
plt.scatter(x1, x2, c=clusters, cmap='viridis', alpha=0.6)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('KMeans Clustering Result (2D)')
plt.colorbar(label='Cluster')
plt.grid(True)

# Step 4: Show the plot
plt.show()
