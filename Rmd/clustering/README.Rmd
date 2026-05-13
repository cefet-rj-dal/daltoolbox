# Clustering Examples

This section organizes clustering by modeling family instead of treating all unsupervised methods as interchangeable. The numbering now leaves visible gaps so the reader can distinguish the introductory partitional/medoid methods from density-based clustering and from tuning-oriented workflows.

A good reading order is to start with centroid-based clustering, compare it with medoid-based clustering, then move to density-based discovery, and only after that inspect clustering model selection.

## Partition-Based Foundations

These examples are the fastest way to understand how `daltoolbox` represents unsupervised learners, cluster assignments, and evaluation outputs.

- [01-partitional-kmeans.md](/examples/clustering/01-partitional-kmeans.md) - `clu_kmeans`: centroid-based clustering with Euclidean partitions.
- [02-medoid-pam.md](/examples/clustering/02-medoid-pam.md) - `clu_pam`: medoid-based clustering, often more robust to extreme points.

## Density-Based Clustering

This block separates methods that detect dense regions from those that force every instance into a centroid or medoid partition.

- [10-density-dbscan.md](/examples/clustering/10-density-dbscan.md) - `clu_dbscan`: groups dense neighborhoods and can leave noise points unassigned.

## Soft, Model-Based, and Hierarchical Views

This block widens the clustering perspective beyond hard partitions. These methods are useful when the analyst wants memberships, probabilistic mixtures, hierarchical merge structure, or graph communities.

- [11-fuzzy-cmeans.md](/examples/clustering/11-fuzzy-cmeans.md) - `cluster_cmeans`: fuzzy clustering with membership degrees.
- [12-model-based-gmm.md](/examples/clustering/12-model-based-gmm.md) - `cluster_gmm`: Gaussian mixture model clustering.
- [13-hierarchical-hclust.md](/examples/clustering/13-hierarchical-hclust.md) - `cluster_hclust`: agglomerative hierarchical clustering.
- [14-graph-louvain.md](/examples/clustering/14-graph-louvain.md) - `cluster_louvain_graph`: graph community detection by modularity optimization.

## Model Selection

This final example is kept apart because it is about choosing configurations, not only understanding a single clustering family.

- [20-model-selection-tuning.md](/examples/clustering/20-model-selection-tuning.md) - `clu_tune`: compares clustering configurations through the DAL tuning workflow.
