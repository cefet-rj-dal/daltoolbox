# Transformations Examples

This section organizes transformations by analytical purpose. The numbering now leaves semantic gaps so the reader can immediately distinguish sampling, cleaning, scaling, encoding, smoothing, balancing, feature selection, and dimensionality-reduction blocks.

The main didactic question is always the same: what is changing in the dataset, and why would that change improve the next stage of the workflow?

## Sampling

These examples are useful when the first challenge is creating representative partitions before any other transformation.

- [01-sampling-random.md](/examples/transf/01-sampling-random.md) - `sample_random`: train/test split and folds by simple random draws.
- [02-sampling-stratified.md](/examples/transf/02-sampling-stratified.md) - `sample_stratified`: train/test split and folds preserving target proportions.
- [03-sampling-simple.md](/examples/transf/03-sampling-simple.md) - `sample_simple`: direct random extraction of rows or vector elements.
- [04-sampling-balance.md](/examples/transf/04-sampling-balance.md) - `sample_balance`: up-sampling or down-sampling through the sampling interface.
- [05-sampling-cluster.md](/examples/transf/05-sampling-cluster.md) - `sample_cluster`: select entire groups defined by a categorical attribute.

## Cleaning and Data Quality

These examples address missing values and unusual observations before the modeling stage begins.

- [10-cleaning-na-removal.md](/examples/transf/10-cleaning-na-removal.md) - remove rows with missing values through `na.omit`.
- [11-cleaning-outliers-boxplot.md](/examples/transf/11-cleaning-outliers-boxplot.md) - detect outliers by the IQR boxplot rule.
- [12-cleaning-outliers-gaussian.md](/examples/transf/12-cleaning-outliers-gaussian.md) - flag outliers through Gaussian distance from the mean.
- [13-cleaning-imputation-tree.md](/examples/transf/13-cleaning-imputation-tree.md) - `imputation_tree`: iterative model-based imputation for mixed data.

## Scaling

These examples make numeric attributes more suitable for downstream learners by changing their scale.

- [20-scaling-minmax.md](/examples/transf/20-scaling-minmax.md) - `minmax`: rescale numeric attributes to a bounded interval.
- [21-scaling-zscore.md](/examples/transf/21-scaling-zscore.md) - `zscore`: standardize numeric attributes by mean and standard deviation.

## Encoding

These examples change the representation of attributes so downstream learners can consume categorical inputs safely.

- [30-encoding-categorical-mapping.md](/examples/transf/30-encoding-categorical-mapping.md) - `categ_mapping`: one-hot encoding for categorical variables.
- [31-encoding-hierarchy-cut.md](/examples/transf/31-encoding-hierarchy-cut.md) - `hierarchy_cut`: create ordered categories from numeric cut points.
- [32-feature-generation.md](/examples/transf/32-feature-generation.md) - `feature_generation`: derive new attributes from existing ones.

## Smoothing and Discretization

These examples summarize continuous values into intervals, frequencies, or cluster-defined bins.

- [40-smoothing-interval.md](/examples/transf/40-smoothing-interval.md) - `smoothing_inter`: discretization by equal-width intervals.
- [41-smoothing-frequency.md](/examples/transf/41-smoothing-frequency.md) - `smoothing_freq`: discretization by frequency-balanced bins.
- [42-smoothing-clustering.md](/examples/transf/42-smoothing-clustering.md) - `smoothing_cluster`: discretization guided by clustering structure.
- [43-smoothing-generic.md](/examples/transf/43-smoothing-generic.md) - `smoothing`: base family and tuning logic behind the smoothing variants.

## Balancing

These examples correct class imbalance after the earlier variable transformations are already defined.

- [50-balancing-oversampling.md](/examples/transf/50-balancing-oversampling.md) - `bal_oversampling`: oversampling by replication or local SMOTE.
- [51-balancing-subsampling.md](/examples/transf/51-balancing-subsampling.md) - `bal_subsampling`: undersampling to reduce majority-class dominance.

## Feature Selection

These examples reduce the predictor set according to relevance, sparsity, or subset search.

- [60-feature-selection-information-gain.md](/examples/transf/60-feature-selection-information-gain.md) - `feature_selection_info_gain`: rank predictors by information gain.
- [61-feature-selection-relief.md](/examples/transf/61-feature-selection-relief.md) - `feature_selection_relief`: rank predictors through nearest-hit and nearest-miss contrasts.
- [62-feature-selection-stepwise.md](/examples/transf/62-feature-selection-stepwise.md) - `feature_selection_stepwise`: GLM-based stepwise selection.
- [63-feature-selection-lasso.md](/examples/transf/63-feature-selection-lasso.md) - `feature_selection_lasso`: L1-based embedded feature selection.
- [64-feature-selection-fss.md](/examples/transf/64-feature-selection-fss.md) - `feature_selection_fss`: forward subset search.

## Dimensionality Reduction and Heuristics

These final examples help the reader decide how many dimensions or cut points may be enough for a problem.

- [70-dimensionality-pca.md](/examples/transf/70-dimensionality-pca.md) - `dt_pca`: principal component analysis for projection and compression.
- [71-curvature-minimum.md](/examples/transf/71-curvature-minimum.md) - `fit_curvature_min`: detect minimum curvature for increasing curves.
- [72-curvature-maximum.md](/examples/transf/72-curvature-maximum.md) - `fit_curvature_max`: detect maximum curvature for decreasing curves.

## Dataset Summaries

These examples restructure the dataset at the group level instead of only modifying single rows or columns.

- [80-aggregation-by-group.md](/examples/transf/80-aggregation-by-group.md) - `aggregation`: grouped summaries through named expressions.
