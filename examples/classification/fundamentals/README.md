# Foundations Examples

These examples establish the minimum logic of a classification experiment: define the target, create a split, fit a learner, generate class scores, and evaluate the result.

> Slide deck: `d042-fundamentals.pdf` covers the baseline and the decision tree (01, 02). It does not yet cover the `rpart` backend (21) — recommend recreating this deck to add `cla_rpart` as an alternate backend for the same tree topic.

- [01-baseline-majority.md](/examples/classification/fundamentals/01-baseline-majority.md) - `cla_majority`: baseline classifier that always predicts the most frequent class observed during training.
- [02-interpretable-tree.md](/examples/classification/fundamentals/02-interpretable-tree.md) - `cla_dtree`: decision tree classifier with an easy-to-interpret splitting structure.
- [21-tree-rpart.md](/examples/classification/fundamentals/21-tree-rpart.md) - `cla_rpart`: CART classification tree with the `rpart` backend.
