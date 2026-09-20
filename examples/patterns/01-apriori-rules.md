About the method
- `pat_apriori`: association-rule mining with semantic configuration stored in the miner object.

Didactic goal: establish the standard pattern-mining line of experiment used throughout this family with a directional rule example: fixed `rhs`, adaptive support/confidence thresholds, and post-mining quality filtering by lift.

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
## transactions as itemMatrix in sparse format with
##  48842 rows (elements/itemsets/transactions) and
##  114 columns (items) and a density of 0.1274938 
## 
## most frequent items:
##       capital-gain=[0,1e+05]    capital-loss=[0,4.36e+03] 
##                        48842                        48842 
## native-country=United-States                   race=White 
##                        43832                        41762 
##       hours-per-week=[40,99]                      (Other) 
##                        37155                       489451 
## 
## element (itemset/transaction) length distribution:
## sizes
##    11    12    13    14    15 
##    19   971  2067 15623 30162 
## 
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##   11.00   14.00   15.00   14.53   15.00   15.00 
## 
## includes extended item information - examples:
##        labels variables  levels
## 1 age=[17,31)       age [17,31)
## 2 age=[31,44)       age [31,44)
## 3 age=[44,90]       age [44,90]
## 
## includes extended transaction information - examples:
##   transactionID
## 1             1
## 2             2
## 3             3
```

Model configuration.

``` r
utils <- patutils()

pm <- pat_apriori(
  target = "rules",
  supp = 0,
  conf = 0,
  support_strategy = pat_support_threshold("curvature"),
  confidence_strategy = pat_confidence_threshold("rhs_baseline", margin = 0.1),
  minlen = 2,
  maxlen = 3,
  rhs = c("native-country=United-States"),
  quality_filter = utils$quality_min(confidence = 0.9, lift = 1.03),
  control = list(verbose = FALSE)
)
```

Fit and discover patterns.

``` r
pm <- fit(pm, trans)
pm$engine_parameter
```

```
## $supp
## [1] 0.03470374
## 
## $minlen
## [1] 2
## 
## $maxlen
## [1] 3
## 
## $target
## [1] "rules"
## 
## $conf
## [1] 0.95
```

``` r
rules <- suppressWarnings(discover(pm, trans))
length(rules)
```

```
## [1] 3
```

Evaluate the discovered patterns.

``` r
eval <- evaluate(pm, rules)
eval$metrics
```

```
##            metric     value      type
## 1   pattern_count 3.0000000 intrinsic
## 2    mean_support 0.1231113 intrinsic
## 3 mean_confidence 0.9516825 intrinsic
## 4       mean_lift 1.0604599 intrinsic
## 5     mean_length 3.0000000 intrinsic
## 6  retained_ratio 1.0000000    filter
```

Inspect a few patterns.

``` r
ord <- order(arules::quality(rules)$lift, arules::quality(rules)$confidence, decreasing = TRUE)
arules::inspect(rules[head(ord, 6)])
```

```
##     lhs                              rhs                               support confidence   coverage     lift count
## [1] {fnlwgt=[1.23e+04,1.41e+05),                                                                                   
##      marital-status=Divorced}     => {native-country=United-States} 0.04481798  0.9521531 0.04707014 1.060984  2189
## [2] {workclass=Local-gov,                                                                                          
##      race=White}                  => {native-country=United-States} 0.04985463  0.9515436 0.05239343 1.060305  2435
## [3] {fnlwgt=[1.23e+04,1.41e+05),                                                                                   
##      race=White}                  => {native-country=United-States} 0.27466115  0.9513510 0.28870644 1.060090 13415
```

What to observe
- The thresholds matter a lot. If `supp` or `conf` is `0`, the corresponding strategy estimates it during `fit()`.
- A fixed `rhs` allows `pat_confidence_threshold("rhs_baseline")` to set confidence relative to the consequent frequency.
- Constraining the `rhs` is useful when you want rules that explain a specific consequent.
- `lift` works as a second filter on top of support and confidence, removing rules that are frequent but not especially informative.
- Later pattern examples will keep this same body and only change the pattern family and configuration.
