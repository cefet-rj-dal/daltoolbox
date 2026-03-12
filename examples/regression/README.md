# Regression Examples

This section reorganizes regression examples as a sequence of modeling ideas. Instead of listing methods only by their class names, the examples are grouped according to the kind of predictive reasoning they illustrate.

For learners, the main question here is: how do numeric prediction workflows stay stable while the modeling assumptions change?

## Interpretable Starting Point

Begin with a learner that is easy to explain and useful for understanding the flow of regression experiments.

- [01-interpretable-tree.md](/examples/regression/01-interpretable-tree.md) - `reg_dtree`: decision tree for regression. Partitions feature space and estimates values by leaf means; interpretable and can model nonlinearities.

## Neighborhood and Ensemble Models

These examples move to methods that either borrow information from nearby observations or aggregate many weak predictors.

- [02-instance-based-knn.md](/examples/regression/02-instance-based-knn.md) - `reg_knn`: k-Nearest Neighbors for regression. Predicts the mean (or weighted mean) of targets from the k nearest neighbors.
- [03-ensemble-random-forest.md](/examples/regression/03-ensemble-random-forest.md) - `reg_rf`: Random Forest for regression. Averages many decision trees trained with randomness; tends to reduce variance.

## Flexible Nonlinear Models

These learners are useful when the relationship between predictors and target is more complex and less easily explained by a single tree.

- [04-margin-svm.md](/examples/regression/04-margin-svm.md) - `reg_svm`: Support Vector Regression (SVR). Models a function with an error-insensitive margin up to `epsilon` and penalizes violations via `cost`.
- [05-neural-mlp.md](/examples/regression/05-neural-mlp.md) - `reg_mlp`: Multilayer Perceptron (neural network) for regression.

## Model Selection

The final example shows how tuning fits into the same regression workflow without changing the underlying experimental logic.

- [06-model-selection-tuning.md](/examples/regression/06-model-selection-tuning.md) - `reg_tune`: hyperparameter search for regression models over ranges in `ranges`.
