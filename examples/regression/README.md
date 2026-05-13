# Regression Examples

This section organizes regression examples as a progression of modeling ideas. The numbering now leaves semantic gaps so the collection is easier to scan by family: interpretable models, local methods, ensembles, margin-based regression, neural learners, and tuning.

If you are learning numeric prediction in `daltoolbox`, read the examples in order. The early notebooks help build intuition before the later ones introduce stronger nonlinear models and hyperparameter search.

## Interpretable Start

- [01-interpretable-tree.md](/examples/regression/01-interpretable-tree.md) - `reg_dtree`: regression tree with rule-like splits and easy inspection.

## Core Regression Families

- [11-linear-model.md](/examples/regression/11-linear-model.md) - `reg_lm`: linear regression baseline for numeric prediction.
- [10-instance-based-knn.md](/examples/regression/10-instance-based-knn.md) - `reg_knn`: local prediction by nearby cases.
- [20-ensemble-random-forest.md](/examples/regression/20-ensemble-random-forest.md) - `reg_rf`: tree ensemble for robust nonlinear regression.
- [30-margin-svm.md](/examples/regression/30-margin-svm.md) - `reg_svm`: support vector regression with margin-based fitting.
- [40-neural-mlp.md](/examples/regression/40-neural-mlp.md) - `reg_mlp`: multilayer perceptron for nonlinear numeric prediction.

## Model Selection

- [50-model-selection-tuning.md](/examples/regression/50-model-selection-tuning.md) - `reg_tune`: searches hyperparameter settings for regression models in a reproducible workflow.
