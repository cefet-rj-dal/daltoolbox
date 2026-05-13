About the method
- `pat_apriori`: association-rule mining with semantic configuration stored in the miner object.

Didactic goal: establish the standard pattern-mining line of experiment used throughout this family: load data, configure the miner, `fit()`, `discover()`, `evaluate()`, and inspect the patterns.

Environment setup.

``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "arules"))

library(daltoolbox)
```

Load transactional data.

``` r
data("AdultUCI", package = "arules")
trans <- suppressWarnings(methods::as(as.data.frame(AdultUCI), "transactions"))
summary(trans)
```

```
##       Length        Class         Mode 
##        48842 transactions           S4
```

Model configuration.

``` r
utils <- patutils()

pm <- pat_apriori(
  target = "rules",
  supp = 0.5,
  conf = 0.9,
  minlen = 2,
  maxlen = 4,
  rhs = c("income=small"),
  quality_filter = utils$quality_min(confidence = 0.95, lift = 1.05)
)
```

Fit and discover patterns.

``` r
pm <- fit(pm, trans)
rules <- discover(pm, trans)
```

```
## Apriori
## 
## Parameter specification:
##  confidence minval smax arem  aval originalSupport maxtime support minlen maxlen target  ext
##         0.9    0.1    1 none FALSE            TRUE       5     0.5      2      4  rules TRUE
## 
## Algorithmic control:
##  filter tree heap memopt load sort verbose
##     0.1 TRUE TRUE  FALSE TRUE    2    TRUE
## 
## Absolute minimum support count: 24421 
## 
## set item appearances ...[1 item(s)] done [0.00s].
## set transactions ...[114 item(s), 48842 transaction(s)] done [0.04s].
## sorting and recoding items ... [9 item(s)] done [0.00s].
## creating transaction tree ... done [0.01s].
## checking subsets of size 1 2 3 4
```

```
## Warning in arules::apriori(data, parameter = obj$engine_parameter, appearance = obj$engine_appearance, : Mining stopped
## (maxlen reached). Only patterns up to a length of 4 returned!
```

```
##  done [0.00s].
## writing ... [0 rule(s)] done [0.00s].
## creating S4 object  ... done [0.00s].
```

``` r
length(rules)
```

```
## [1] 0
```

Evaluate the discovered patterns.

``` r
eval <- evaluate(pm, rules)
eval$metrics
```

```
##            metric value      type
## 1   pattern_count     0 intrinsic
## 2    mean_support   NaN intrinsic
## 3 mean_confidence   NaN intrinsic
## 4       mean_lift   NaN intrinsic
## 5     mean_length    NA intrinsic
## 6  retained_ratio    NA    filter
```

Inspect a few patterns.

``` r
if (length(rules) == 0) {
  cat("No rules remained after the quality filter. Lower the thresholds if you want to inspect some candidates.\n")
} else {
  arules::inspect(rules[seq_len(min(6, length(rules)))])
}
```

```
## No rules remained after the quality filter. Lower the thresholds if you want to inspect some candidates.
```

What to observe
- The standard pattern-mining workflow is different from prediction, but it is still structured and reproducible.
- Later pattern examples will keep this same body and only change the pattern family and configuration.
