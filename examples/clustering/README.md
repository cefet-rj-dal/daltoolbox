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

## Model Selection

This final example is kept apart because it is about choosing configurations, not only understanding a single clustering family.

- [20-model-selection-tuning.md](/examples/clustering/20-model-selection-tuning.md) - `clu_tune`: compares clustering configurations through the DAL tuning workflow.
