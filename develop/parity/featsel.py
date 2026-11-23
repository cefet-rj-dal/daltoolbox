"""Python side of the feature-selection R-parity check (same rows).

Each selector exposes ``.selected`` after ``fit``; parity is the agreement of the
selected feature SET with R on the identical rows. Regression-style selectors run on
diabetes; the binomial stepwise and ReliefF run on breast_cancer. Exports the full
datasets to CSV so ``featsel_samesplit.R`` scores the very same rows.

Run::

    python daltoolbox/develop/parity/featsel.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def _breast() -> pd.DataFrame:
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df = df.drop(columns="target")
    df["target"] = pd.Categorical.from_codes(raw.target, raw.target_names)
    return df


def _sel(name, model, data):
    model.fit(data)
    print(f"PY {name:22s} -> {sorted(model.selected)}")


def main() -> None:
    diabetes = dal.datasets.diabetes()
    breast = _breast()
    diabetes.to_csv(HERE / "diabetes_full.csv", index=False)
    breast.to_csv(HERE / "breast_full.csv", index=False)

    print("# diabetes (regression-style selectors)")
    _sel("corr(0.9)", dal.FeatureSelectionCorr(cutoff=0.9), diabetes)
    _sel("lasso", dal.FeatureSelectionLasso("target"), diabetes)
    _sel("fss", dal.FeatureSelectionFss("target"), diabetes)
    _sel("stepwise(gauss)", dal.FeatureSelectionStepwise("target", direction="forward", family="gaussian"), diabetes)

    print("# breast_cancer (classification selectors)")
    _sel("stepwise(fwd,binom)", dal.FeatureSelectionStepwise("target", direction="forward"), breast)
    _sel("relief(top10)", dal.FeatureSelectionRelief("target", top=10, m=50, k=10, seed=1), breast)
    _sel("infogain(top10)", dal.FeatureSelectionInfoGain("target", top=10), breast)
    print("DONE")


if __name__ == "__main__":
    main()
