# R ↔ Python parity verification

This directory holds the scripts that verify, family by family, that the Python port
[`pydaltoolbox`](https://github.com/cefet-rj-dal/pydaltoolbox) reproduces this R package
(`daltoolbox`) — and, where they disagreed, that led to fixes **on both sides**.

The full methodology (why we do it, the taxonomy of disagreements, how to localize a
defect, threats to validity) is written up in the port repository at
`pydaltoolbox/develop/parity.md`. This README records the R-side specifics.

## Method in one paragraph

Comparing two implementations is only meaningful on **byte-for-byte identical inputs**.
For each family we freeze the input (and, for supervised cases, one split) to CSV in one
language and have **both** sides read the same file; then we compare the decision-level
invariant (test metric / selected set / partition agreement / rule set / full transformed
matrix). Any gap is classified as *exact*, *benign* (stochastic / numeric / structural),
or a *real defect* to be fixed.

## Scripts

Each family has a Python producer (freezes the input + runs the Python side) and an R
`*_samesplit.R` consumer (reads the same CSV + runs the R side). Run the producer first,
then the R script (from this repo root):

| Family            | Python producer                 | R consumer                          |
|-------------------|---------------------------------|-------------------------------------|
| Regression        | (split exported once)           | `regression_diabetes_sameplit.R`    |
| Classification    | `classification_breast.py`      | `classification_breast_samesplit.R` |
| Clustering        | `clustering_iris.py`            | `clustering_iris_samesplit.R`        |
| Feature selection | `featsel.py`                    | `featsel_samesplit.R`                |
| Pattern mining    | `patterns_apriori.py`           | `patterns_apriori_samesplit.R`       |
| Transformations   | `transforms.py`                 | `transforms_samesplit.R` (+ `transforms_compare.py`) |

```sh
# example (classification)
python develop/parity/classification_breast.py        # writes breast_*.csv
Rscript --vanilla develop/parity/classification_breast_samesplit.R
```

Run R scripts with `Rscript --vanilla` (per the project convention); they `load_all()`
this package. The `out_*.csv` produced by the transformation comparison are intermediate
and are git-ignored.

## Results (and the fixes that landed in this R package)

Six families verified. Most disagreements were benign (explained with evidence); the rest
were genuine defects. **Three fixes landed here, on the R side** — parity testing improved
the original package, not only the port:

- **`cla_glm.R` — prediction inversion (correctness bug).** The binomial GLM models
  `P(Y = last factor level)`, but `predict` assigned that probability to the `positive`
  class column unconditionally, inverting predictions whenever `positive` was not the last
  level (breast-cancer accuracy 0.04 instead of 0.96). Latent for years because the
  roxygen example happened to put `positive` last. Fixed by releveling the response in
  `fit.cla_glm`.
- **`clu_kmeans.R` — weak default.** `stats::kmeans` was left at `nstart = 1`, whose single
  random start can land in a poor local optimum (~0.67 purity on iris instead of ~0.89).
  Added an `nstart` argument (default 10), matching the robust multi-start the port uses.
- **`trans_categ_mapping.R` — output convention.** `categ_mapping` returned only the dummy
  columns (named `Speciessetosa`); it now keeps the other columns and replaces the
  attribute in place by `<attribute>_<level>` dummies, matching the port and making it
  safe inside a pipeline.

Defects fixed on the **port** side (no R change needed, R was the reference): decision-tree
pre-pruning (`mindev`), DBSCAN `eps` k-distance off-by-one and its elbow detector
(now reproducing `smooth.spline(df = 2)`), and PCA standardization (`prcomp`'s sample-SD,
`ddof = 1`). Benign-by-nature disagreements (documented, not fixed): CV-`lambda` lasso and
binomial stepwise under quasi-separation; and `mlxtend`'s multi-item-consequent association
rules, which `arules::apriori` does not emit by default.

The canonical back-implementations (PAM, c-means, correlation filter, forward selection,
ReliefF, SMOTE, DARA, PrefixSpan) are maintained in the Python port and backported here;
their roxygen cites the original paper and notes the backport relationship.
