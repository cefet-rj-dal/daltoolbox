# Transformations Examples

This section introduces transformation and data-preparation examples, which are often the first step of an Experiment Line. The objective is to show how raw data can be cleaned, normalized, discretized, sampled, balanced, and enriched before modeling.

These examples are a good starting point for new readers because they make the workflow logic visible without the additional complexity of a predictive model. They also show that data preparation in `daltoolbox` follows the same pattern of defining an object, fitting when necessary, transforming the data, and moving to the next stage of the analysis.

If you want a simple progression, start with `na_removal`, `normalization_minmax`, and `normalization_zscore`, then explore sampling and balancing, and finally move to feature selection, smoothing, PCA, and curvature-based examples.

- [categorical_mapping.md](/examples/transf/categorical_mapping.md) - `categ_mapping`: converts a categorical column into binary variables (one-hot). Can use n columns or n-1 columns.
- [curvature_maximum.md](/examples/transf/curvature_maximum.md) - `fit_curvature_max`: computes curvature via the second derivative of a smoothed spline and returns the maximum curvature position for decreasing curves; useful to choose a trade-off point where further reductions add little benefit.
- [curvature_minimum.md](/examples/transf/curvature_minimum.md) - `fit_curvature_min`: computes curvature via the second derivative of a smoothed spline over the sequence and returns the minimum curvature position for increasing curves; useful to find a trade-off point where additional gains become marginal.
- [sample_oversampling.md](/examples/transf/sample_oversampling.md) - `bal_oversampling`: class oversampling with random replication or local SMOTE.
- [sample_subsampling.md](/examples/transf/sample_subsampling.md) - `bal_subsampling`: random class undersampling to the minority count.
- [dal_pca.md](/examples/transf/dal_pca.md) - `dt_pca`: Principal Component Analysis (PCA) projects correlated variables onto orthogonal components ordered by explained variance. You can let the tool pick the number of components via an elbow heuristic or set it explicitly.
- [dal_smoothing_clustering.md](/examples/transf/dal_smoothing_clustering.md) - `smoothing_cluster`: discretization/smoothing by defining bins via clustering instead of fixed intervals.
- [dal_smoothing_frequency.md](/examples/transf/dal_smoothing_frequency.md) - `smoothing_freq`: discretization/smoothing by frequency (quantiles), producing bins with similar counts.
- [dal_smoothing_interval.md](/examples/transf/dal_smoothing_interval.md) - `smoothing_inter`: discretization/smoothing by regular intervals (equal widths). Useful to summarize continuous variables into ranges.
- [feature_selection_fss.md](/examples/transf/feature_selection_fss.md) - `feature_selection_fss`: forward stepwise subset search for numeric targets.
- [feature_selection_information_gain.md](/examples/transf/feature_selection_information_gain.md) - `feature_selection_info_gain`: rank predictors by information gain.
- [feature_selection_lasso.md](/examples/transf/feature_selection_lasso.md) - `feature_selection_lasso`: L1-based embedded feature selection for numeric targets.
- [feature_selection_relief.md](/examples/transf/feature_selection_relief.md) - `feature_selection_relief`: rank predictors by nearest-hit and nearest-miss comparisons.
- [feature_selection_stepwise.md](/examples/transf/feature_selection_stepwise.md) - `feature_selection_stepwise`: stepwise GLM-based selection for classification.
- [na_removal.md](/examples/transf/na_removal.md) - NA removal: use `na.omit` to drop instances with missing values. Useful for initial cleanup when imputation is not desired.
- [normalization_minmax.md](/examples/transf/normalization_minmax.md) - `minmax`: linearly rescales numeric attributes to a target range (default [0, 1]). Useful for scale-sensitive algorithms and models that expect bounded inputs.
- [normalization_zscore.md](/examples/transf/normalization_zscore.md) - `zscore`: standardizes numeric attributes to zero mean and unit variance. You can also rescale to a target mean (`nmean`) and standard deviation (`nsd`).
- [outliers_boxplot.md](/examples/transf/outliers_boxplot.md) - `outliers_boxplot`: identifies outliers by the boxplot rule (Q1 - 1.5*IQR, Q3 + 1.5*IQR) and can remove them from numeric attributes.
- [outliers_gaussian.md](/examples/transf/outliers_gaussian.md) - `outliers_gaussian`: flags as outliers values beyond mean +/- 3 standard deviations, assuming approximately normal distribution.
- [sample_random.md](/examples/transf/sample_random.md) - `sample_random`: splits train/test sets and creates folds via random draws, preserving only expected proportions on average.
- [sample_stratified.md](/examples/transf/sample_stratified.md) - `sample_stratified`: splits train/test and folds preserving the target variable proportion (stratification) per category.

