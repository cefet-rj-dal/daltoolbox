# daltoolbox Examples

The documentation is organized to support two complementary entry
points:

- a guided tutorial track for readers who want to learn the workflow
  step by step
- thematic example collections for readers who want to inspect a
  specific family of methods

If you are new to `daltoolbox`, start with the tutorials. If you
already know the package structure, the thematic collections remain
available and were reorganized with more didactic descriptions.

### Guided tutorial track

- [Tutorials](https://github.com/cefet-rj-dal/daltoolbox/tree/main/examples/tutorial/)
  - a 13-part learning sequence covering first experiment, sampling,
  data quality, preprocessing, baselines, metrics, model comparison,
  tuning, end-to-end pipelines, regression, clustering, visual
  analysis, and custom extensions.

### Thematic example collections

- [Transformations](https://github.com/cefet-rj-dal/daltoolbox/tree/main/examples/transf/)
  - sampling, balancing, cleaning, scaling, encoding, smoothing,
  feature selection, dimensionality reduction, and curvature-based
  heuristics.
- [Classification](https://github.com/cefet-rj-dal/daltoolbox/tree/main/examples/classification/)
  - baseline models, core classifier families, and model selection.
- [Regression](https://github.com/cefet-rj-dal/daltoolbox/tree/main/examples/regression/)
  - interpretable models, nonlinear learners, and tuning for numeric
  prediction.
- [Clustering](https://github.com/cefet-rj-dal/daltoolbox/tree/main/examples/clustering/)
  - partitional, medoid-based, density-based methods, and clustering
  model selection.
- [Graphics](https://github.com/cefet-rj-dal/daltoolbox/tree/main/examples/graphics/)
  - comparison, distribution, relationship, time-oriented, and
  export-focused visualizations.
- [Custom extensions](https://github.com/cefet-rj-dal/daltoolbox/tree/main/examples/custom/)
  - examples showing how to integrate new transformations, classifiers,
  regressors, and clustering methods into the Experiment Line workflow.

### Documentation design

The examples were revised to be more useful for learning:

- files inside each collection are now numbered in a suggested reading
  order
- category `README` files group examples by subject rather than only by
  class name
- many examples now include more explanation between code blocks,
  including interpretation hints and common mistakes
