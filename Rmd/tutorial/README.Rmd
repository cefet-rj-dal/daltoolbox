# Tutorial Track

This section presents `daltoolbox` as a guided learning path. The numbering now leaves semantic gaps so the reader can distinguish preparation tutorials, classification-analysis tutorials, end-to-end workflow transfer, and closing topics such as visual reporting and extension.

The sequence is cumulative. Each tutorial introduces one main decision in a data mining study, explains why that step matters, and keeps the code close to that learning objective.

## Foundations

Start here if you want the minimum workflow before comparing learners or tuning parameters.

- [01-first-experiment.md](/examples/tutorial/01-first-experiment.md) - understand the core Experiment Line cycle: split, fit, predict, evaluate.
- [02-sampling-strategy.md](/examples/tutorial/02-sampling-strategy.md) - see how train/test splitting and folds affect the quality of an experiment.
- [03-data-quality-and-cleaning.md](/examples/tutorial/03-data-quality-and-cleaning.md) - inspect missing values, outliers, and categorical transformations before modeling.
- [04-preprocessing-basics.md](/examples/tutorial/04-preprocessing-basics.md) - normalize data and reduce redundant variables.

## Classification Study Cycle

This block keeps the reader inside one supervised task while varying the analytical question: baseline, metrics, comparison, and tuning.

- [10-baseline-classification.md](/examples/tutorial/10-baseline-classification.md) - start with a baseline before trying stronger learners.
- [11-metrics-and-evaluation.md](/examples/tutorial/11-metrics-and-evaluation.md) - read classification results with more care and connect metrics to model behavior.
- [12-model-comparison.md](/examples/tutorial/12-model-comparison.md) - compare different learners under the same split and evaluation protocol.
- [13-tuning-workflow.md](/examples/tutorial/13-tuning-workflow.md) - use hyperparameter search in a controlled way.

## Workflow Transfer

These tutorials show how the same experiment logic extends to complete pipelines and to other learning tasks.

- [20-end-to-end-classification-pipeline.md](/examples/tutorial/20-end-to-end-classification-pipeline.md) - connect preparation, modeling, and evaluation in one complete classification flow.
- [21-regression-workflow.md](/examples/tutorial/21-regression-workflow.md) - transfer the same workflow to numeric prediction.
- [22-clustering-workflow.md](/examples/tutorial/22-clustering-workflow.md) - move to unsupervised learning and study the role of normalization.

## Communication and Extension

Finish with examples that broaden the workflow toward reporting and framework customization.

- [30-visual-analysis-and-reporting.md](/examples/tutorial/30-visual-analysis-and-reporting.md) - use plots to understand data and communicate results.
- [31-custom-extension.md](/examples/tutorial/31-custom-extension.md) - create a custom learner while preserving the same workflow structure.
