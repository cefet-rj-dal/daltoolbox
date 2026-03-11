# Custom Examples

Examples focused on how to extend `daltoolbox` with custom components while preserving the same Experiment Line workflow.

- [custom_transformation.md](custom_transformation.md) - `smote_custom`: shows how to create a custom transformation with a constructor and `transform()`, using `smotefamily::SMOTE` as the concrete example.
- [custom_classification.md](custom_classification.md) - `cla_rsnns_custom`: shows how to create a custom classifier with constructor, `fit()`, and `predict()`, using `RSNNS::mlp`.
- [custom_regression.md](custom_regression.md) - `reg_rsnns_custom`: shows how to create a custom regressor with constructor, `fit()`, and `predict()`, using `RSNNS::mlp` with linear output.
- [custom_clustering.md](custom_clustering.md) - `cluster_agnes_custom`: shows how to create a custom clusterer with constructor, `fit()`, and `cluster()`, using `cluster::agnes`.
