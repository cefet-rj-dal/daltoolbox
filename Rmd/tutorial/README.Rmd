# Tutorial Track

This section reorganizes `daltoolbox` as a guided learning path. Instead of presenting only isolated functions, the tutorials below walk through the main decisions that appear in a data mining study: understanding the workflow, splitting data, cleaning and transforming attributes, building baselines, reading metrics, comparing models, tuning, integrating complete pipelines, and extending the framework.

The sequence is cumulative. Each tutorial introduces one main objective, explains why that step matters in a mining project, and then shows the code with more context than the reference examples. The goal is to help both readers who are new to `daltoolbox` and readers who are still learning the logic of data mining experiments.

Recommended reading order:

- [01-first-experiment.md](/examples/tutorial/01-first-experiment.md) - understand the core Experiment Line cycle: split, fit, predict, evaluate.
- [02-sampling-strategy.md](/examples/tutorial/02-sampling-strategy.md) - see how train/test splitting and folds affect the quality of an experiment.
- [03-data-quality-and-cleaning.md](/examples/tutorial/03-data-quality-and-cleaning.md) - inspect missing values, outliers, and categorical transformations before modeling.
- [04-preprocessing-basics.md](/examples/tutorial/04-preprocessing-basics.md) - normalize data and reduce redundant variables.
- [05-baseline-classification.md](/examples/tutorial/05-baseline-classification.md) - start with a baseline before trying stronger learners.
- [06-metrics-and-evaluation.md](/examples/tutorial/06-metrics-and-evaluation.md) - read classification results with more care and connect metrics to model behavior.
- [07-model-comparison.md](/examples/tutorial/07-model-comparison.md) - compare different learners under the same split and evaluation protocol.
- [08-tuning-workflow.md](/examples/tutorial/08-tuning-workflow.md) - use hyperparameter search in a controlled way.
- [09-end-to-end-classification-pipeline.md](/examples/tutorial/09-end-to-end-classification-pipeline.md) - connect preparation, modeling, and evaluation in one complete classification flow.
- [10-regression-workflow.md](/examples/tutorial/10-regression-workflow.md) - transfer the same workflow to numeric prediction.
- [11-clustering-workflow.md](/examples/tutorial/11-clustering-workflow.md) - move to unsupervised learning and study the role of normalization.
- [12-visual-analysis-and-reporting.md](/examples/tutorial/12-visual-analysis-and-reporting.md) - use plots to understand data and communicate results.
- [13-custom-extension.md](/examples/tutorial/13-custom-extension.md) - create a custom learner while preserving the same workflow structure.
