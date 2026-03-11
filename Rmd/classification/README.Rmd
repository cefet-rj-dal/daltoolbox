# Classification Examples

This section introduces supervised classification examples built around the standard `daltoolbox` workflow. The objective is to show how different classifiers can be configured, trained, used for prediction, and evaluated without changing the overall structure of the analysis.

These examples are useful for readers who want to compare alternative learning strategies for categorical targets while keeping the same data handling and evaluation logic. They range from simple baselines to more expressive nonlinear models and hyperparameter search.

If you want a simple entry point, start with `cla_majority` or `cla_dtree`, then move to `cla_knn`, `cla_nb`, and `cla_rf`, and finish with `cla_mlp`, `cla_svm`, and `cla_tune`.

- [cla_dtree.md](/examples/classification/cla_dtree.md) — `cla_dtree`: Decision Tree for classification. Recursively splits on explanatory variables to separate classes, yielding an interpretable model.
- [cla_knn.md](/examples/classification/cla_knn.md) — `cla_knn`: k-Nearest Neighbors classifier. Classifies by majority vote among the k nearest neighbors in feature space.
- [cla_majority.md](/examples/classification/cla_majority.md) — `cla_majority`: baseline classifier that always predicts the most frequent class observed during training. Useful as a minimum performance reference.
- [cla_mlp.md](/examples/classification/cla_mlp.md) — `cla_mlp`: Multilayer Perceptron (feedforward neural network) for classification.
- [cla_nb.md](/examples/classification/cla_nb.md) — `cla_nb`: Naive Bayes for classification. Probabilistic model assuming conditional independence among features; simple, efficient, and often competitive.
- [cla_rf.md](/examples/classification/cla_rf.md) — `cla_rf`: Random Forest for classification. Ensemble of decision trees trained with randomness; robust and handles heterogeneous features well.
- [cla_svm.md](/examples/classification/cla_svm.md) — `cla_svm`: Support Vector Machine for classification, maximizing the margin between classes.
- [cla_tune.md](/examples/classification/cla_tune.md) — `cla_tune`: performs hyperparameter search for a classifier over ranges defined in `ranges`.

