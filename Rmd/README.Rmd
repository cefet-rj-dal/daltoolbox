# daltoolbox Examples

This directory contains the source R Markdown examples used to illustrate how `daltoolbox` supports data preparation, modeling, visualization, and customization in a consistent Experiment Line workflow.

The main goal of this collection is not only to show isolated functions, but to help readers understand how complete analytical tasks can be structured with a uniform interface. Across the examples, the same ideas appear repeatedly: prepare the data, configure a method, fit the object, apply the action expected for that object, and evaluate the result when appropriate.

If you are new to the package, a good path is to start with the transformation examples, then move to classification or regression, and only after that explore clustering, graphics, and custom extensions. This progression makes it easier to understand how the examples build on the same design principles.

The links below point to the generated Markdown files under `examples/`, which are the rendered outputs intended for reading.

- [Tutorials](/examples/tutorial/README.md) - 13 tutorials organized as a guided path from first workflow to custom extension.
- [Classification](/examples/classification/README.md) - 8 examples (e.g., cla_dtree, cla_knn, cla_majority). Supervised classification algorithms and tuning.
- [Clustering](/examples/clustering/README.md) - 4 examples (e.g., clu_dbscan, clu_kmeans, clu_pam). Unsupervised clustering methods and model selection.
- [Custom](/examples/custom/README.md) - 4 examples (e.g., custom_transformation, custom_classification, custom_clustering). Didactic examples showing how to integrate external methods into the Experiment Line workflow.
- [Graphics](/examples/graphics/README.md) - 17 examples (e.g., grf_bar_error, grf_bar, grf_boxplot). Chart examples with ggplot2 and helpers.
- [Regression](/examples/regression/README.md) - 6 examples (e.g., reg_dtree, reg_knn, reg_mlp). Supervised regression algorithms and tuning.
- [Transformations](/examples/transf/README.md) - 21 examples including discretization, normalization, curvature analysis, feature selection, and balancing.

