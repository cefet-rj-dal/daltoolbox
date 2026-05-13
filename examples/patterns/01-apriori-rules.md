About the method
- `pat_apriori`: association-rule mining with semantic configuration in the object. Instead of sending raw engine lists, the user configures support, confidence, length, left-hand side, right-hand side, and optional quality filters as properties of the miner.

Didactic goal: read this example as a descriptive workflow. The value is not in predicting a target, but in understanding how the configuration of the search space changes the rules that survive.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# Pattern mining - apriori rules

# installation
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

Configure the miner with semantic properties instead of raw engine arguments.

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

Fit the miner and discover rules.

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
## set transactions ...[114 item(s), 48842 transaction(s)] done [0.03s].
## sorting and recoding items ... [9 item(s)] done [0.00s].
## creating transaction tree ... done [0.01s].
## checking subsets of size 1 2 3 4
```

```
## Warning in arules::apriori(data, parameter = obj$engine_parameter, appearance = obj$engine_appearance, : Mining stopped
## (maxlen reached). Only patterns up to a length of 4 returned!
```

```
##  done [0.03s].
## writing ... [0 rule(s)] done [0.00s].
## creating S4 object  ... done [0.00s].
```

``` r
length(rules)
```

```
## [1] 0
```

Evaluate the discovered rules using the metrics configured by the object family.

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

Inspect a few rules.

``` r
arules::inspect(rules[1:min(6, length(rules))])
```

```
## Error in `h()`:
## ! error in evaluating the argument 'x' in selecting a method for function 'inspect': subscript out of bounds
```

What to observe
- `fit()` records the schema and compiles the engine configuration from semantic properties such as `supp`, `conf`, `lhs`, and `rhs`.
- `discover()` applies the mining engine and then optional quality filtering.
- `evaluate()` summarizes how many rules survived and how strong they are on average.

Common mistakes
- Treating `confidence` as if it were a predictive guarantee rather than a descriptive conditional frequency.
- Using a very restrictive `rhs` together with high `supp` and high `conf`, which may eliminate almost all rules.
- Interpreting large rule sets without any post-filtering.
