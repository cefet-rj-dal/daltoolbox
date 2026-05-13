## Pattern Miner Base

This example documents `pattern_miner` as the common contract behind the pattern-discovery learners. Unlike the algorithm-specific examples, the goal here is not to discover patterns, but to inspect the slots and workflow expectations shared by `pat_apriori`, `pat_eclat`, and `pat_cspade`.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```


``` r
miner <- pattern_miner()
class(miner)
```

```
## [1] "pattern_miner" "dal_learner"   "dal_base"
```

``` r
names(miner)
```

```
## [1] "fitted"       "pat_utils"    "eval_metrics" "pattern_kind"
```

``` r
miner$pattern_kind
```

```
## [1] "patterns"
```

What to observe
- Pattern miners revolve around `fit()`, `discover()`, and `evaluate()`.
- Algorithm-specific constructors mostly specialize filtering, engine configuration, and quality summaries on top of this base object.
