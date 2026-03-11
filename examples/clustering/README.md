# Clustering Examples

This section presents unsupervised clustering examples in `daltoolbox`. The objective is to help readers understand how grouping methods can be configured, fitted, converted into cluster labels, and externally evaluated when a reference label is available.

Because clustering is exploratory by nature, these examples are especially useful for understanding how the package separates the clustering step itself from the evaluation step. They also show how different notions of structure can be captured, such as compact groups, density-connected regions, and medoid-based partitions.

A good reading order is `clu_kmeans`, then `clu_pam`, then `clu_dbscan`, and finally `clu_tune` to see how hyperparameter selection can also be organized within the same workflow.

- [clu_dbscan.md](/examples/clustering/clu_dbscan.md) — `cluster_dbscan`: density-based method. Identifies dense regions separated by sparse areas; detects noise and arbitrarily shaped clusters.
- [clu_kmeans.md](/examples/clustering/clu_kmeans.md) — `cluster_kmeans`: partitions data into k groups by minimizing within-cluster variance. Sensitive to scale; normalization can improve results.
- [clu_pam.md](/examples/clustering/clu_pam.md) — `cluster_pam`: Partitioning Around Medoids. Similar to k-means but uses medoids (real points) instead of centroids, making it more robust to outliers.
- [clu_tune.md](/examples/clustering/clu_tune.md) — `clu_tune`: selects hyperparameters for a clustering method. In this example, it chooses `k` for `cluster_kmeans` over a range.

