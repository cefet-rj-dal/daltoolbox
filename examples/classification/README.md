# Classification Examples

This section now presents classification as a learning path rather than as a flat list of model names. The goal is to help the reader progress from simple baselines to stronger learners and, only after that, to model selection.

The examples below still work as references, but the grouping is designed to answer a more practical question: what should I study first if I am learning supervised classification in `daltoolbox`?

## Foundations

These examples establish the minimum logic of a classification experiment: create a split, fit a learner, generate predictions, and evaluate the outcome.

- [01-baseline-majority.md](/examples/classification/01-baseline-majority.md) - `cla_majority`: baseline classifier that always predicts the most frequent class observed during training. Useful as a minimum performance reference.
- [02-interpretable-tree.md](/examples/classification/02-interpretable-tree.md) - `cla_dtree`: Decision Tree for classification. Recursively splits on explanatory variables to separate classes, yielding an interpretable model.

## Core Model Families

After the foundations, these examples show alternative modeling ideas: local neighborhoods, probabilistic assumptions, ensembles, margins, and neural networks.

- [03-instance-based-knn.md](/examples/classification/03-instance-based-knn.md) - `cla_knn`: k-Nearest Neighbors classifier. Classifies by majority vote among the k nearest neighbors in feature space.
- [04-probabilistic-naive-bayes.md](/examples/classification/04-probabilistic-naive-bayes.md) - `cla_nb`: Naive Bayes for classification. Probabilistic model assuming conditional independence among features; simple, efficient, and often competitive.
- [05-ensemble-random-forest.md](/examples/classification/05-ensemble-random-forest.md) - `cla_rf`: Random Forest for classification. Ensemble of decision trees trained with randomness; robust and handles heterogeneous features well.
- [06-margin-svm.md](/examples/classification/06-margin-svm.md) - `cla_svm`: Support Vector Machine for classification, maximizing the margin between classes.
- [07-neural-mlp.md](/examples/classification/07-neural-mlp.md) - `cla_mlp`: Multilayer Perceptron (feedforward neural network) for classification.

## Model Selection

This final example shows how `daltoolbox` organizes hyperparameter search without breaking the same Experiment Line structure used by the previous learners.

- [08-model-selection-tuning.md](/examples/classification/08-model-selection-tuning.md) - `cla_tune`: performs hyperparameter search for a classifier over ranges defined in `ranges`.
