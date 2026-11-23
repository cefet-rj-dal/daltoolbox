"""Python side of the transformation R-parity check (deterministic transforms).

Transforms are deterministic, so parity is the strongest form: the transformed output
MATRIX must equal R's up to a tiny tolerance on the identical input. Each transform's
output is written to ``out_py_<name>.csv``; ``transforms_samesplit.R`` writes
``out_r_<name>.csv``; ``transforms_compare.py`` diffs the pairs.

Run::

    python daltoolbox/develop/parity/transforms.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def main() -> None:
    iris = dal.datasets.iris()
    iris.to_csv(HERE / "iris_full.csv", index=False)

    # a copy with a fixed NaN pattern for NA/imputation transforms
    iris_na = iris.copy()
    for col, rows in {"sepal_length": [0, 10], "petal_width": [5, 15, 25]}.items():
        iris_na.loc[rows, col] = np.nan
    iris_na.to_csv(HERE / "iris_na.csv", index=False)

    def dump(name, df):
        df.to_csv(HERE / f"out_py_{name}.csv", index=False)
        print(f"PY  {name:22s} shape={df.shape}")

    dump("minmax", dal.MinMax().fit(iris).transform(iris))
    dump("zscore", dal.ZScore().fit(iris).transform(iris))
    dump("categ_mapping", dal.CategMapping("Species").fit(iris).transform(iris))

    pca = dal.DtPca("Species", components=2).fit(iris)
    out = pca.transform(iris)
    num = out.select_dtypes("number").abs()  # PCA sign is arbitrary -> compare |value|
    dump("dt_pca_abs", pd.concat([num, out[["Species"]]], axis=1))

    dump("na_removal", dal.NaRemoval().fit(iris_na).transform(iris_na))
    dump("imputation_median", dal.ImputationSimple(method="median").fit(iris_na).transform(iris_na))
    dump("outliers_boxplot", dal.OutliersBoxplot().fit(iris).transform(iris))
    dump("outliers_gaussian", dal.OutliersGaussian().fit(iris).transform(iris))
    print("DONE")


if __name__ == "__main__":
    main()
