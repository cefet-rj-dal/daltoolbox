
<!-- README.md is generated from README.Rmd. Please edit that file -->

# <img src='https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/master/inst/logo.png' align='centre' height='150' width='139'/> DAL Toolbox

<!-- badges: start -->

![GitHub Repo
stars](https://img.shields.io/github/stars/cefet-rj-dal/daltoolbox?logo=Github)
![CRAN downloads](https://cranlogs.r-pkg.org/badges/daltoolbox)
<!-- badges: end -->

DAL Toolbox is an R framework for data analytics workflows. It organizes
preprocessing, modeling, evaluation, tuning, visualization, and
extensibility resources around a consistent Experiment Line style API,
helping users build reproducible end-to-end analytical pipelines.

The package supports data preprocessing, classification, regression,
clustering, pattern mining, graphics, and time series prediction. It is
designed for teaching, experimentation, and applied data science
projects where the same workflow needs to be reused, compared, and
extended across methods.

Current package version in this repository: `1.3.767`.

------------------------------------------------------------------------

## Installation

The stable version is available on CRAN:

<https://CRAN.R-project.org/package=daltoolbox>

``` r
install.packages("daltoolbox")
```

The development version is available on GitHub:

<https://github.com/cefet-rj-dal/daltoolbox>

``` r
library(devtools)
devtools::install_github("cefet-rj-dal/daltoolbox", force = TRUE, dependencies = FALSE, upgrade = "never")
```

------------------------------------------------------------------------

## Documentation

Documentation and examples are available in the package site and in the
repository:

- [Package website](https://cefet-rj-dal.github.io/daltoolbox/)
- [Examples](https://github.com/cefet-rj-dal/daltoolbox/tree/main/examples/)
- [GitHub repository](https://github.com/cefet-rj-dal/daltoolbox)
- [JOSS
  paper](https://github.com/cefet-rj-dal/daltoolbox/blob/main/paper/paper.pdf)

The documentation is organized around two complementary entry points:

- a guided tutorial track for readers who want to learn the workflow
  step by step
- thematic example collections for readers who want to inspect a
  specific family of methods

If you are new to `daltoolbox`, start with the tutorials. If you already
know the package structure, the thematic collections provide focused
examples by method family.

------------------------------------------------------------------------

## Guided Tutorial Track

The tutorials are part of the
[examples](https://github.com/cefet-rj-dal/daltoolbox/tree/main/examples/)
collection. They form a 13-part learning sequence covering first
experiment, sampling, data quality, preprocessing, baselines, metrics,
model comparison, tuning, end-to-end pipelines, regression, clustering,
visual analysis, and custom extensions.

The sequence is cumulative. Each tutorial introduces one main decision
in a data mining study, explains why that step matters, and keeps the
code close to that learning objective.

------------------------------------------------------------------------

## Thematic Example Collections

The
[examples](https://github.com/cefet-rj-dal/daltoolbox/tree/main/examples/)
collection includes thematic subcollections:

- Transformations - sampling, balancing, cleaning, scaling, encoding,
  smoothing, feature selection, dimensionality reduction, and
  curvature-based heuristics.
- Classification - baseline models, decision trees, instance-based
  methods, probabilistic models, linear models, ensembles, neural
  models, support vector machines, boosting, and tuning.
- Regression - interpretable models, instance-based learners, random
  forests, support vector machines, neural models, and tuning for
  numeric prediction.
- Clustering - partitional, medoid-based, density-based, fuzzy,
  model-based, hierarchical, graph-based methods, and clustering model
  selection.
- Pattern Mining - association rules, frequent itemsets, and sequence
  mining.
- Graphics - comparison, distribution, relationship, time-oriented, and
  export-focused visualizations.
- Custom Extensions - examples showing how to integrate new
  transformations, classifiers, regressors, clusterers, autoencoders,
  and pattern miners into the Experiment Line workflow.

------------------------------------------------------------------------

## Main Capabilities

- Unified abstractions for learners, transformations, predictors, and
  tuners.
- Reusable preprocessing workflows for sampling, balancing, cleaning,
  normalization, encoding, smoothing, feature selection, and
  dimensionality reduction.
- Classification, regression, clustering, and pattern mining examples
  with consistent fit, predict, evaluate, and tune stages.
- Time series support for preprocessing, augmentation, normalization,
  filtering, and prediction.
- Visualization helpers for comparisons, distributions, relationships,
  time series, and report-oriented graphics.
- Extensible interfaces for custom analytical components.

------------------------------------------------------------------------

## Related DAL Projects

- [DAL Toolbox website](https://cefet-rj-dal.github.io/daltoolbox/)
- [tspredit](https://cefet-rj-dal.github.io/tspredit/)
- [harbinger](https://cefet-rj-dal.github.io/harbinger/)
- [Data Analytics Lab](https://eic.cefet-rj.br/~dal/)

------------------------------------------------------------------------

## Playlist

[DAL Toolbox
videos](https://www.youtube.com/playlist?list=PLG1M6TA-XJo8)

[![Watch the playlist on
YouTube](https://img.shields.io/badge/YouTube-Watch%20playlist-red?logo=youtube&logoColor=white)](https://www.youtube.com/playlist?list=PLG1M6TA-XJo8)

------------------------------------------------------------------------

## Bugs and Feature Requests

Please report bugs, questions, and feature requests at:

<https://github.com/cefet-rj-dal/daltoolbox/issues>
