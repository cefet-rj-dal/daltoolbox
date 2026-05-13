# Classification Examples

This section presents classification as a staged learning path rather than as a flat list of model names. The numbering now leaves semantic gaps so the reader can immediately see where the collection moves from foundations to core model families and then to model selection.

If you are new to supervised classification in `daltoolbox`, start with the first two examples and only then move to the alternative learner families. The final block is intentionally separated because tuning makes more sense after the reader already understands at least one untuned learner.

## Foundations

These examples establish the minimum logic of a classification experiment: define the target, create a split, fit a learner, generate class scores, and evaluate the result.

- [01-baseline-majority.md](/examples/classification/01-baseline-majority.md) - `cla_majority`: baseline classifier that always predicts the most frequent class observed during training.
- [02-interpretable-tree.md](/examples/classification/02-interpretable-tree.md) - `cla_dtree`: decision tree classifier with an easy-to-interpret splitting structure.

## Core Model Families

This block groups learners by different modeling ideas so the reader can compare neighborhood-based, probabilistic, ensemble, margin-based, and neural approaches without confusing them with the baseline block above.

- [10-instance-based-knn.md](/examples/classification/10-instance-based-knn.md) - `cla_knn`: classifies by the majority vote of the nearest neighbors.
- [11-probabilistic-naive-bayes.md](/examples/classification/11-probabilistic-naive-bayes.md) - `cla_nb`: probabilistic classifier based on conditional independence assumptions.
- [12-ensemble-random-forest.md](/examples/classification/12-ensemble-random-forest.md) - `cla_rf`: ensemble of randomized decision trees.
- [13-margin-svm.md](/examples/classification/13-margin-svm.md) - `cla_svm`: support vector machine for margin-based class separation.
- [14-neural-mlp.md](/examples/classification/14-neural-mlp.md) - `cla_mlp`: multilayer perceptron for nonlinear classification.

## Model Selection

This final block isolates hyperparameter search from the learner-introduction examples. That separation helps the reader first understand what a learner does before adding the extra layer of search strategy and comparison.

- [20-model-selection-tuning.md](/examples/classification/20-model-selection-tuning.md) - `cla_tune`: searches hyperparameter ranges while preserving the same Experiment Line workflow.
