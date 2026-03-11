# Regression Examples

This section presents supervised regression examples for numeric targets. The objective is to show how different predictive models can be trained and compared within the same Experiment Line structure, from simpler interpretable models to more flexible nonlinear learners.

These examples are especially helpful for readers who want to understand how `daltoolbox` organizes regression experiments in a uniform way: split data, define the learner, fit it, generate predictions, and evaluate the results with the same logic across models.

A practical reading path is to begin with `reg_dtree`, then explore `reg_knn` and `reg_rf`, and finally move to `reg_mlp`, `reg_svm`, and `reg_tune`.

- [reg_dtree.md](/examples/regression/reg_dtree.md) — `reg_dtree`: decision tree for regression. Partitions feature space and estimates values by leaf means; interpretable and can model nonlinearities.
- [reg_knn.md](/examples/regression/reg_knn.md) — `reg_knn`: k-Nearest Neighbors for regression. Predicts the mean (or weighted mean) of targets from the k nearest neighbors.
- [reg_mlp.md](/examples/regression/reg_mlp.md) — `reg_mlp`: Multilayer Perceptron (neural network) for regression.
- [reg_rf.md](/examples/regression/reg_rf.md) — `reg_rf`: Random Forest for regression. Averages many decision trees trained with randomness; tends to reduce variance.
- [reg_svm.md](/examples/regression/reg_svm.md) — `reg_svm`: Support Vector Regression (SVR). Models a function with an error-insensitive margin up to `epsilon` and penalizes violations via `cost`.
- [reg_tune.md](/examples/regression/reg_tune.md) — `reg_tune`: hyperparameter search for regression models over ranges in `ranges`.

