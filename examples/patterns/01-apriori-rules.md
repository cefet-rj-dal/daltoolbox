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
```

```
## Error in `patutils()`:
## ! could not find function "patutils"
```

``` r
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

```
## Error in `pat_apriori()`:
## ! unused arguments (target = "rules", supp = 0.5, conf = 0.9, minlen = 2, maxlen = 4, rhs = c("income=small"), quality_filter = utils$quality_min(confidence = 0.95, lift = 1.05))
```

Fit and discover patterns.

``` r
pm <- fit(pm, trans)
```

```
## Error:
## ! object 'pm' not found
```

``` r
rules <- discover(pm, trans)
```

```
## Error:
## ! object 'pm' not found
```

``` r
length(rules)
```

```
## Error:
## ! object 'rules' not found
```

Evaluate the discovered patterns.

``` r
eval <- evaluate(pm, rules)
```

```
## Error:
## ! object 'pm' not found
```

``` r
eval$metrics
```

```
## Error in `eval$metrics`:
## ! object of type 'closure' is not subsettable
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
## Error:
## ! object 'rules' not found
```

What to observe
- The standard pattern-mining workflow is different from prediction, but it is still structured and reproducible.
- Later pattern examples will keep this same body and only change the pattern family and configuration.
