"""Compare the transformation outputs written by transforms.py and
transforms_samesplit.R: for each ``out_py_<name>.csv`` / ``out_r_<name>.csv`` pair,
report shape agreement and the maximum absolute numeric difference (and exact match for
non-numeric columns), up to a small tolerance.

Run (after both producers):

    python daltoolbox/develop/parity/transforms_compare.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TOL = 1e-9
NAMES = [
    "minmax", "zscore", "categ_mapping", "dt_pca_abs",
    "na_removal", "imputation_median", "outliers_boxplot", "outliers_gaussian",
]


def compare(name: str) -> str:
    py = pd.read_csv(HERE / f"out_py_{name}.csv")
    r = pd.read_csv(HERE / f"out_r_{name}.csv")
    if py.shape != r.shape:
        return f"SHAPE  py={py.shape} r={r.shape}"
    # align columns by name where possible, else by position
    if set(py.columns) == set(r.columns):
        r = r[py.columns]
    else:
        r.columns = py.columns
    num = py.select_dtypes("number").columns
    max_diff = 0.0
    for c in num:
        a = pd.to_numeric(py[c], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(r[c], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.abs(a - b)
        d[both_nan] = 0.0
        max_diff = max(max_diff, float(np.nanmax(d)) if len(d) else 0.0)
    # non-numeric columns must match exactly
    obj_ok = all(
        (py[c].astype(str).to_numpy() == r[c].astype(str).to_numpy()).all()
        for c in py.columns.difference(num)
    )
    verdict = "OK " if (max_diff <= TOL and obj_ok) else "DIFF"
    return f"{verdict} max_abs_diff={max_diff:.2e} nonnumeric_match={obj_ok}"


def main() -> None:
    for name in NAMES:
        try:
            print(f"{name:22s} {compare(name)}")
        except FileNotFoundError as exc:
            print(f"{name:22s} MISSING {exc}")


if __name__ == "__main__":
    main()
