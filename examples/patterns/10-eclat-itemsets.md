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

pm <- pat_eclat(
  supp = 0,
  support_strategy = pat_support_threshold("curvature"),
  maxlen = 3,
  include = c("sex=Male", "income=small", "marital-status=Married-civ-spouse", "race=White"),
  exclude = c("income=small"),
  quality_filter = utils$quality_min(support = 0.4),
  control = list(verbose = FALSE)
)
```

```
## Error in `pat_eclat()`:
## ! não foi possível encontrar a função "pat_eclat"
```

Fit and discover patterns.

``` r
pm <- fit(pm, trans)
```

```
## Error:
## ! objeto 'pm' não encontrado
```

``` r
pm$engine_parameter
```

```
## Error:
## ! objeto 'pm' não encontrado
```

``` r
itemsets <- discover(pm, trans)
```

```
## Error:
## ! objeto 'pm' não encontrado
```

``` r
length(itemsets)
```

```
## Error:
## ! objeto 'itemsets' não encontrado
```

Evaluate the discovered patterns.

``` r
eval <- evaluate(pm, itemsets)
```

```
## Error:
## ! objeto 'pm' não encontrado
```

``` r
eval$metrics
```

```
## Error in `eval$metrics`:
## ! objeto de tipo 'closure' não possível dividir em subconjuntos
```

Inspect a few patterns.

``` r
ord <- order(arules::quality(itemsets)$support, decreasing = TRUE)
```

```
## Error in `h()`:
## ! erro na avaliação do argumento 'x' na seleção do método para a função 'quality': 'objeto 'itemsets' não encontrado'
```

``` r
arules::inspect(itemsets[ord])
```

```
## Error in `h()`:
## ! erro na avaliação do argumento 'x' na seleção do método para a função 'inspect': 'objeto 'itemsets' não encontrado'
```

What to observe
- `include` and `exclude` act as semantic filters over which items may appear in the result.
- The quality filter is a second stage: it keeps only the itemsets that are still strong after the structural filtering.
- Itemsets are not directional rules. They capture co-occurrence, not implication.
