# Transformations Examples

Data preparation and transformation utilities.

- [categorical_mapping.md](categorical_mapping.md) — `categ_mapping`: converts a categorical column into binary variables (one-hot). Can use n columns or n-1 columns.
- [curvature_maximum.md](curvature_maximum.md) — `fit_curvature_max`: computes curvature via the second derivative of a smoothed spline and returns the maximum curvature position for decreasing curves; useful to choose a trade-off point where further reductions add little benefit.
- [curvature_minimum.md](curvature_minimum.md) — `fit_curvature_min`: computes curvature via the second derivative of a smoothed spline over the sequence and returns the minimum curvature position for increasing curves; useful to find a trade-off point where additional gains become marginal.
- [dal_pca.md](dal_pca.md) — `dt_pca`: Principal Component Analysis (PCA) projects correlated variables onto orthogonal components ordered by explained variance. You can let the tool pick the number of components via an elbow heuristic or set it explicitly.
- [dal_smoothing_clustering.md](dal_smoothing_clustering.md) — `smoothing_cluster`: discretization/smoothing by defining bins via clustering instead of fixed intervals.
- [dal_smoothing_frequency.md](dal_smoothing_frequency.md) — `smoothing_freq`: discretization/smoothing by frequency (quantiles), producing bins with similar counts.
- [dal_smoothing_interval.md](dal_smoothing_interval.md) — `smoothing_inter`: discretization/smoothing by regular intervals (equal widths). Useful to summarize continuous variables into ranges.
- [na_removal.md](na_removal.md) — NA removal: use `na.omit` to drop instances with missing values. Useful for initial cleanup when imputation is not desired.
- [normalization_minmax.md](normalization_minmax.md) — `minmax`: linearly rescales numeric attributes to a target range (default [0, 1]). Useful for scale-sensitive algorithms and models that expect bounded inputs.
- [normalization_zscore.md](normalization_zscore.md) — `zscore`: standardizes numeric attributes to zero mean and unit variance. You can also rescale to a target mean (`nmean`) and standard deviation (`nsd`).
- [outliers_boxplot.md](outliers_boxplot.md) — `outliers_boxplot`: identifies outliers by the boxplot rule (Q1 - 1.5*IQR, Q3 + 1.5*IQR) and can remove them from numeric attributes.
- [outliers_gaussian.md](outliers_gaussian.md) — `outliers_gaussian`: flags as outliers values beyond mean +/- 3 standard deviations, assuming approximately normal distribution.
- [sample_random.md](sample_random.md) — `sample_random`: splits train/test sets and creates folds via random draws, preserving only expected proportions on average.
- [sample_stratified.md](sample_stratified.md) — `sample_stratified`: splits train/test and folds preserving the target variable proportion (stratification) per category.

