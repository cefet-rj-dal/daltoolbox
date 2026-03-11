# Custom Examples

This section is dedicated to one of the most important ideas in `daltoolbox`: extending the Experiment Line workflow with custom components while preserving the same overall structure.

The objective here is didactic. These examples show that integrating a new transformation, classifier, regressor, or clusterer does not require changing the framework itself. Instead, the customization is centered on a small and predictable contract: define an object, store its configuration, and implement the expected methods.

These are good examples for readers who want to understand the extensibility of the package rather than just its built-in methods. A natural reading order is transformation, classification, regression, and then clustering.

- [custom_transformation.md](/examples/custom/custom_transformation.md) - `smote_custom`: shows how to create a custom transformation with a constructor and `transform()`, using `smotefamily::SMOTE` as the concrete example.
- [custom_classification.md](/examples/custom/custom_classification.md) - `cla_rsnns_custom`: shows how to create a custom classifier with constructor, `fit()`, and `predict()`, using `RSNNS::mlp`.
- [custom_regression.md](/examples/custom/custom_regression.md) - `reg_rsnns_custom`: shows how to create a custom regressor with constructor, `fit()`, and `predict()`, using `RSNNS::mlp` with linear output.
- [custom_clustering.md](/examples/custom/custom_clustering.md) - `cluster_agnes_custom`: shows how to create a custom clusterer with constructor, `fit()`, and `cluster()`, using `cluster::agnes`.
