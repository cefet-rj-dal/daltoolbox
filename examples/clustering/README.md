# Clustering Examples

Unsupervised clustering methods and model selection.

- [clu_dbscan.md](clu_dbscan.md) — `cluster_dbscan`: density-based method. Identifies dense regions separated by sparse areas; detects noise and arbitrarily shaped clusters.
- [clu_kmeans.md](clu_kmeans.md) — `cluster_kmeans`: partitions data into k groups by minimizing within-cluster variance. Sensitive to scale; normalization can improve results.
- [clu_pam.md](clu_pam.md) — `cluster_pam`: Partitioning Around Medoids. Similar to k-means but uses medoids (real points) instead of centroids, making it more robust to outliers.
- [clu_tune.md](clu_tune.md) — `clu_tune`: selects hyperparameters for a clustering method. In this example, it chooses `k` for `cluster_kmeans` over a range.

