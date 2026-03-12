# Clustering Examples

This section presents clustering as a guided sequence of unsupervised ideas. The examples are grouped by how they define structure in the data, which makes the progression easier to follow for someone learning exploratory analysis.

The key didactic shift in clustering is that there is no target variable during fitting. What changes from one example to the next is the notion of similarity or grouping imposed by the method.

## Partitional Methods

These examples create clusters by dividing the dataset into a fixed number of groups.

- [01-partitional-kmeans.md](/examples/clustering/01-partitional-kmeans.md) - `cluster_kmeans`: partitions data into k groups by minimizing within-cluster variance. Sensitive to scale; normalization can improve results.
- [02-medoid-pam.md](/examples/clustering/02-medoid-pam.md) - `cluster_pam`: Partitioning Around Medoids. Similar to k-means but uses medoids (real points) instead of centroids, making it more robust to outliers.

## Density-Based Structure

This example shows a different idea of grouping: dense regions separated by sparse space, with the possibility of noise points.

- [03-density-dbscan.md](/examples/clustering/03-density-dbscan.md) - `cluster_dbscan`: density-based method. Identifies dense regions separated by sparse areas; detects noise and arbitrarily shaped clusters.

## Model Selection

The last example shows that even in unsupervised learning, hyperparameters such as the number of clusters can be explored systematically.

- [04-model-selection-tuning.md](/examples/clustering/04-model-selection-tuning.md) - `clu_tune`: selects hyperparameters for a clustering method. In this example, it chooses `k` for `cluster_kmeans` over a range.
