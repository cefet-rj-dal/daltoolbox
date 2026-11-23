"""Python side of the classification R-parity check (breast_cancer, same split).

Exports the IDENTICAL train/test split to CSV so the R script
``classification_breast_samesplit.R`` scores the very same rows, then prints the
Python accuracies. Dataset: scikit-learn ``breast_cancer`` -- harder than iris
(binary, but features span wildly different scales), which exposes scale- and
default-faithfulness bugs that iris (all features in cm) hides.

Run::

    python daltoolbox/develop/parity/classification_breast.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

import pydaltoolbox as dal

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent


def load() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df = df.drop(columns="target")
    df["target"] = pd.Categorical.from_codes(data.target, data.target_names)
    return df


def main() -> None:
    df = load()
    slevels = list(df["target"].cat.categories)
    split = dal.SampleRandom(seed=1).train_test(df)
    train, test = split["train"], split["test"]

    train.to_csv(HERE / "breast_train.csv", index=False)
    test.to_csv(HERE / "breast_test.csv", index=False)

    def acc(model) -> float:
        model.fit(train)
        pred = model.predict(test.drop(columns="target"))
        ev = model.evaluate(test["target"], pred)
        return round(float(ev["metrics"]["accuracy"].iloc[0]), 3)

    models = {
        "ClaMajority": dal.ClaMajority("target", slevels),
        "ClaDtree": dal.ClaDtree("target", slevels),
        "ClaKnn(k5)": dal.ClaKnn("target", slevels, k=5),
        "ClaNb": dal.ClaNb("target", slevels),
        "ClaGlm": dal.ClaGlm("target", slevels),
        "ClaRf(30,5)": dal.ClaRf("target", slevels, ntree=30, mtry=5),
        "ClaMlp(5,.1)": dal.ClaMlp("target", slevels, size=5, decay=0.1),
        "ClaSvm(def)": dal.ClaSvm("target", slevels),
    }
    for name, model in models.items():
        try:
            print(f"PY {name:14s} acc= {acc(model):.3f}")
        except Exception as exc:  # noqa: BLE001
            print(f"PY {name:14s} ERR {type(exc).__name__}: {exc}")
    print("DONE")


if __name__ == "__main__":
    main()
