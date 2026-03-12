# Transformations Examples

This section now organizes transformations by analytical purpose. That change matters because learners usually think in terms of tasks such as sampling, cleaning, scaling, encoding, smoothing, selection, and dimensionality reduction, not in terms of isolated function names.

The question to keep in mind while reading is always the same: what is changing in the dataset, and why would that change improve the next stage of the workflow?

## Sampling and Balance

These examples are useful when the first challenge is creating representative partitions or correcting class imbalance.

- [01-sampling-random.md](/examples/transf/01-sampling-random.md) - `sample_random`: splits train/test sets and creates folds via random draws, preserving only expected proportions on average.
- [02-sampling-stratified.md](/examples/transf/02-sampling-stratified.md) - `sample_stratified`: splits train/test and folds preserving the target variable proportion (stratification) per category.
- [03-balancing-oversampling.md](/examples/transf/03-balancing-oversampling.md) - `bal_oversampling`: class oversampling with random replication or local SMOTE.
- [04-balancing-subsampling.md](/examples/transf/04-balancing-subsampling.md) - `bal_subsampling`: random class undersampling to the minority count.

## Cleaning and Data Quality

These examples address missing values and unusual observations before the modeling stage begins.

- [05-cleaning-na-removal.md](/examples/transf/05-cleaning-na-removal.md) - NA removal: use `na.omit` to drop instances with missing values. Useful for initial cleanup when imputation is not desired.
- [06-cleaning-outliers-boxplot.md](/examples/transf/06-cleaning-outliers-boxplot.md) - `outliers_boxplot`: identifies outliers by the boxplot rule (Q1 - 1.5*IQR, Q3 + 1.5*IQR) and can remove them from numeric attributes.
- [07-cleaning-outliers-gaussian.md](/examples/transf/07-cleaning-outliers-gaussian.md) - `outliers_gaussian`: flags as outliers values beyond mean +/- 3 standard deviations, assuming approximately normal distribution.

## Scaling and Encoding

These examples make attributes more suitable for downstream learners by changing scale or representation.

- [08-scaling-minmax.md](/examples/transf/08-scaling-minmax.md) - `minmax`: linearly rescales numeric attributes to a target range (default [0, 1]). Useful for scale-sensitive algorithms and models that expect bounded inputs.
- [09-scaling-zscore.md](/examples/transf/09-scaling-zscore.md) - `zscore`: standardizes numeric attributes to zero mean and unit variance. You can also rescale to a target mean (`nmean`) and standard deviation (`nsd`).
- [10-encoding-categorical-mapping.md](/examples/transf/10-encoding-categorical-mapping.md) - `categ_mapping`: converts a categorical column into binary variables (one-hot). Can use n columns or n-1 columns.

## Smoothing and Discretization

These examples summarize continuous values into intervals, frequencies, or cluster-defined bins.

- [11-smoothing-interval.md](/examples/transf/11-smoothing-interval.md) - `smoothing_inter`: discretization/smoothing by regular intervals (equal widths). Useful to summarize continuous variables into ranges.
- [12-smoothing-frequency.md](/examples/transf/12-smoothing-frequency.md) - `smoothing_freq`: discretization/smoothing by frequency (quantiles), producing bins with similar counts.
- [13-smoothing-clustering.md](/examples/transf/13-smoothing-clustering.md) - `smoothing_cluster`: discretization/smoothing by defining bins via clustering instead of fixed intervals.

## Feature Selection

These examples reduce the predictor set according to relevance, sparsity, or subset search.

- [14-feature-selection-information-gain.md](/examples/transf/14-feature-selection-information-gain.md) - `feature_selection_info_gain`: rank predictors by information gain.
- [15-feature-selection-relief.md](/examples/transf/15-feature-selection-relief.md) - `feature_selection_relief`: rank predictors by nearest-hit and nearest-miss comparisons.
- [16-feature-selection-stepwise.md](/examples/transf/16-feature-selection-stepwise.md) - `feature_selection_stepwise`: stepwise GLM-based selection for classification.
- [17-feature-selection-lasso.md](/examples/transf/17-feature-selection-lasso.md) - `feature_selection_lasso`: L1-based embedded feature selection for numeric targets.
- [18-feature-selection-fss.md](/examples/transf/18-feature-selection-fss.md) - `feature_selection_fss`: forward stepwise subset search for numeric targets.

## Dimensionality Reduction and Heuristics

These final examples help the reader decide how many dimensions or cut points may be enough for a problem.

- [19-dimensionality-pca.md](/examples/transf/19-dimensionality-pca.md) - `dt_pca`: Principal Component Analysis (PCA) projects correlated variables onto orthogonal components ordered by explained variance.
- [20-curvature-minimum.md](/examples/transf/20-curvature-minimum.md) - `fit_curvature_min`: computes curvature via the second derivative of a smoothed spline over the sequence and returns the minimum curvature position for increasing curves.
- [21-curvature-maximum.md](/examples/transf/21-curvature-maximum.md) - `fit_curvature_max`: computes curvature via the second derivative of a smoothed spline and returns the maximum curvature position for decreasing curves.
