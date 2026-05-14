About the method
- `pat_eclat`: frequent itemset mining. The goal is to identify groups of items that often appear together.

Didactic goal: show item-level filtering explicitly. Here the mining threshold defines the candidate pool, `include`/`exclude` restrict the item vocabulary, and the quality filter keeps only the strongest surviving itemsets.

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
##       capital-gain=[0,1e+05]    capital-loss=[0,4.36e+03] native-country=United-States                   race=White 
##                        48842                        48842                        43832                        41762 
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

pm <- pat_eclat(
  supp = 0.2,
  maxlen = 3,
  include = c("sex=Male", "income=small", "marital-status=Married-civ-spouse", "race=White"),
  exclude = c("income=small"),
  quality_filter = utils$quality_min(support = 0.4),
  control = list(verbose = FALSE)
)
```

Fit and discover patterns.

``` r
pm <- fit(pm, trans)
itemsets <- discover(pm, trans)
length(itemsets)
```

```
## [1] 6
```

Evaluate the discovered patterns.

``` r
eval <- evaluate(pm, itemsets)
eval$metrics
```

```
##           metric     value      type
## 1  pattern_count 6.0000000 intrinsic
## 2   mean_support 0.5646745 intrinsic
## 3    mean_length 1.5000000 intrinsic
## 4 retained_ratio 0.0122449    filter
```

Inspect a few patterns.

``` r
ord <- order(arules::quality(itemsets)$support, decreasing = TRUE)
arules::inspect(itemsets[ord])
```

```
##     items                                           support   count
## [1] {race=White}                                    0.8550428 41762
## [2] {sex=Male}                                      0.6684820 32650
## [3] {race=White, sex=Male}                          0.5883256 28735
## [4] {marital-status=Married-civ-spouse}             0.4581917 22379
## [5] {marital-status=Married-civ-spouse, race=White} 0.4105892 20054
## [6] {marital-status=Married-civ-spouse, sex=Male}   0.4074157 19899
```

What to observe
- `include` and `exclude` act as semantic filters over which items may appear in the result.
- The quality filter is a second stage: it keeps only the itemsets that are still strong after the structural filtering.
- Itemsets are not directional rules. They capture co-occurrence, not implication.
