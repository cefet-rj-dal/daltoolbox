"""Python side of the clustering R-parity check (iris, same rows).

Clustering is unsupervised, so parity is measured as agreement of the partition with
the known reference labels (iris ``Species``): external entropy (lower = better) and
purity (higher = better), computed on the IDENTICAL rows the R script
``clustering_iris_samesplit.R`` reads from ``clu_iris.csv``.

Run::

    python daltoolbox/develop/parity/clustering_iris.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pydaltoolbox as dal
from pydaltoolbox.clustering.metrics import metric_entropy, metric_purity

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent


def main() -> None:
    iris = dal.datasets.iris()
    iris.to_csv(HERE / "clu_iris.csv", index=False)
    x = iris.drop(columns="Species")
    ref = iris["Species"]

    def score(model) -> str:
        model.fit(x)
        clu = model.cluster(x)
        ent = metric_entropy(cluster=clu, attribute=ref)["value"]
        pur = metric_purity(cluster=clu, attribute=ref)["value"]
        return f"entropy= {ent:.3f}  purity= {pur:.3f}"

    models = {
        "Kmeans(k3)": dal.ClusterKmeans(k=3, seed=1),
        "Pam(k3)": dal.ClusterPam(k=3),
        "Cmeans(c3)": dal.ClusterCmeans(centers=3, seed=1),
        "Gmm(g3)": dal.ClusterGmm(g=3, seed=1),
        "Hclust(k3)": dal.ClusterHclust(k=3),
        "Dbscan": dal.ClusterDbscan(min_pts=3),
    }
    for name, model in models.items():
        try:
            print(f"PY {name:12s} {score(model)}")
        except Exception as exc:  # noqa: BLE001
            print(f"PY {name:12s} ERR {type(exc).__name__}: {exc}")
    print("DONE")


if __name__ == "__main__":
    main()
