"""Python side of the pattern-mining R-parity check (apriori, same transactions).

Runs PatApriori on a small one-hot basket with known frequent itemsets and prints the
itemsets (with support) and the rules (support/confidence/lift), so the R script
``patterns_apriori_samesplit.R`` mines the very same transactions with arules.

Sequences (PrefixSpan) and DARA were already validated identical to cSPADE and to the
paper/R in earlier work; this closes the remaining apriori parity.

Run::

    python daltoolbox/develop/parity/patterns_apriori.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def main() -> None:
    basket = pd.DataFrame(
        {
            "milk": [1, 1, 0, 1, 1],
            "bread": [1, 1, 1, 1, 0],
            "butter": [0, 1, 0, 1, 1],
            "beer": [0, 0, 1, 0, 0],
        }
    )
    basket.to_csv(HERE / "basket.csv", index=False)

    miner = dal.PatApriori(target="frequent itemsets", supp=0.4, minlen=1, maxlen=3).fit(basket)
    itemsets = miner.discover(basket)
    print("# frequent itemsets (supp>=0.4)")
    for s, sup in sorted(
        zip(itemsets["itemsets"], itemsets["support"]),
        key=lambda t: (len(t[0]), sorted(t[0])),
    ):
        print(f"PY  {{{','.join(sorted(s))}}}  supp={round(float(sup), 3)}")

    rules = dal.PatApriori(target="rules", supp=0.4, conf=0.6, minlen=2).fit(basket).discover(basket)
    print("# rules (supp>=0.4, conf>=0.6)")
    for _, r in rules.sort_values(["support", "confidence"], ascending=False).iterrows():
        lhs = ",".join(sorted(r["antecedents"]))
        rhs = ",".join(sorted(r["consequents"]))
        print(
            f"PY  {{{lhs}}} => {{{rhs}}}  "
            f"supp={round(float(r['support']), 3)} conf={round(float(r['confidence']), 3)} "
            f"lift={round(float(r['lift']), 3)}"
        )
    print("DONE")


if __name__ == "__main__":
    main()
