# Cleaning and Data Quality Examples

These examples address missing values and unusual observations before the modeling stage begins.

> Slide deck: [d033_transf_cleaning.pdf](https://github.com/cefet-rj-dal/daltoolbox/blob/main/examples/pdf/d033_transf_cleaning.pdf).

- [10-cleaning-na-removal.md](/examples/transf/cleaning/10-cleaning-na-removal.md) - remove rows with missing values through `na.omit`.
- [11-cleaning-outliers-boxplot.md](/examples/transf/cleaning/11-cleaning-outliers-boxplot.md) - detect outliers by the IQR boxplot rule.
- [12-cleaning-outliers-gaussian.md](/examples/transf/cleaning/12-cleaning-outliers-gaussian.md) - flag outliers through Gaussian distance from the mean.
- [13-cleaning-imputation-tree.md](/examples/transf/cleaning/13-cleaning-imputation-tree.md) - `imputation_tree`: tree-based predictive imputation for one target column.
