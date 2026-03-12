# Custom Examples

This section now organizes custom examples by extension role. That makes the progression more natural for learning because the reader first sees how to extend preprocessing, then supervised learners, and only after that unsupervised components.

The unifying idea is always the same: `daltoolbox` keeps the workflow stable while the user plugs in a new backend through a small integration contract.

## Custom Data Preparation

Start here to see the lightest kind of extension: transforming the data before modeling.

- [01-custom-transformation.md](/examples/custom/01-custom-transformation.md) - `smote_custom`: shows how to create a custom transformation with a constructor and `transform()`, using `smotefamily::SMOTE` as the concrete example.

## Custom Supervised Learners

These examples show how to integrate custom classifiers and regressors while preserving the same `fit()` and prediction logic seen in built-in examples.

- [02-custom-classification.md](/examples/custom/02-custom-classification.md) - `cla_rsnns_custom`: shows how to create a custom classifier with constructor, `fit()`, and `predict()`, using `RSNNS::mlp`.
- [03-custom-regression.md](/examples/custom/03-custom-regression.md) - `reg_rsnns_custom`: shows how to create a custom regressor with constructor, `fit()`, and `predict()`, using `RSNNS::mlp` with linear output.

## Custom Unsupervised Components

Finish here to see how the same integration contract extends to clustering.

- [04-custom-clustering.md](/examples/custom/04-custom-clustering.md) - `cluster_agnes_custom`: shows how to create a custom clusterer with constructor, `fit()`, and `cluster()`, using `cluster::agnes`.
