About the method
- `pat_eclat`: frequent itemset mining. Unlike association rules, the result is not directional; the goal is to identify groups of items that often appear together.

Didactic goal: read this example as a structural co-occurrence analysis. The important point is how support, maximum length, and item filters control the combinatorial search.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# Pattern mining - eclat itemsets

# installation
# install.packages(c("daltoolbox", "arules"))

library(daltoolbox)
```

Load the same transactional dataset used for rule mining.

``` r
data("AdultUCI", package = "arules")
adult_df <- as.data.frame(AdultUCI)
head(adult_df)
```

```
##   age        workclass fnlwgt education education-num     marital-status        occupation  relationship  race    sex
## 1  39        State-gov  77516 Bachelors            13      Never-married      Adm-clerical Not-in-family White   Male
## 2  50 Self-emp-not-inc  83311 Bachelors            13 Married-civ-spouse   Exec-managerial       Husband White   Male
## 3  38          Private 215646   HS-grad             9           Divorced Handlers-cleaners Not-in-family White   Male
## 4  53          Private 234721      11th             7 Married-civ-spouse Handlers-cleaners       Husband Black   Male
## 5  28          Private 338409 Bachelors            13 Married-civ-spouse    Prof-specialty          Wife Black Female
## 6  37          Private 284582   Masters            14 Married-civ-spouse   Exec-managerial          Wife White Female
##   capital-gain capital-loss hours-per-week native-country income
## 1         2174            0             40  United-States  small
## 2            0            0             13  United-States  small
## 3            0            0             40  United-States  small
## 4            0            0             40  United-States  small
## 5            0            0             40           Cuba  small
## 6            0            0             40  United-States  small
```

Configure the miner with item filters and quality filtering.

``` r
utils <- patutils()

pm <- pat_eclat(
  supp = 0.5,
  maxlen = 3,
  include = c("sex=Male", "income=small", "marital-status=Married-civ-spouse"),
  quality_filter = utils$quality_min(support = 0.55)
)
```

Fit and discover itemsets. Here the `fit()` step accepts a regular data frame and normalizes it internally to transactions.

``` r
pm <- fit(pm, adult_df)
```

```
## Warning: Column(s) 1, 3, 5, 11, 12, 13 not logical or factor. Applying default discretization (see '? discretizeDF').
```

```
## Warning in discretize(x = c(2174L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 14084L, 5178L, : The calculated breaks are: 0, 0, 0, 99999
##   Only unique breaks are used reducing the number of intervals. Look at ? discretize for details.
```

```
## Warning in discretize(x = c(0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, : The calculated breaks are: 0, 0, 0, 4356
##   Only unique breaks are used reducing the number of intervals. Look at ? discretize for details.
```

```
## Warning in discretize(x = c(40L, 13L, 40L, 40L, 40L, 40L, 16L, 45L, 50L, : The calculated breaks are: 1, 40, 40, 99
##   Only unique breaks are used reducing the number of intervals. Look at ? discretize for details.
```

``` r
itemsets <- discover(pm, adult_df)
```

```
## Warning: Column(s) 1, 3, 5, 11, 12, 13 not logical or factor. Applying default discretization (see '? discretizeDF').
```

```
## Warning in discretize(x = c(2174L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 14084L, 5178L, : The calculated breaks are: 0, 0, 0, 99999
##   Only unique breaks are used reducing the number of intervals. Look at ? discretize for details.
```

```
## Warning in discretize(x = c(0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, : The calculated breaks are: 0, 0, 0, 4356
##   Only unique breaks are used reducing the number of intervals. Look at ? discretize for details.
```

```
## Warning in discretize(x = c(40L, 13L, 40L, 40L, 40L, 40L, 16L, 45L, 50L, : The calculated breaks are: 1, 40, 40, 99
##   Only unique breaks are used reducing the number of intervals. Look at ? discretize for details.
```

```
## Eclat
## 
## parameter specification:
##  tidLists support minlen maxlen            target  ext
##     FALSE     0.5      1      3 frequent itemsets TRUE
## 
## algorithmic control:
##  sparse sort verbose
##       7   -2    TRUE
## 
## Absolute minimum support count: 24421 
## 
## create itemset ... 
## set transactions ...[114 item(s), 48842 transaction(s)] done [0.03s].
## sorting and recoding items ... [9 item(s)] done [0.00s].
## creating bit matrix ... [9 row(s), 48842 column(s)] done [0.00s].
## writing  ... [61 set(s)] done [0.00s].
## Creating S4 object  ... done [0.00s].
```

``` r
length(itemsets)
```

```
## [1] 1
```

Evaluate the itemsets.

``` r
eval <- evaluate(pm, itemsets)
eval$metrics
```

```
##           metric      value      type
## 1  pattern_count 1.00000000 intrinsic
## 2   mean_support 0.66848204 intrinsic
## 3    mean_length 1.00000000 intrinsic
## 4 retained_ratio 0.01639344    filter
```

Inspect a few itemsets.

``` r
arules::inspect(itemsets[1:min(6, length(itemsets))])
```

```
##     items      support  count
## [1] {sex=Male} 0.668482 32650
```

What to observe
- `include` and `exclude` act as DAL-level filters over the discovered itemsets.
- This makes the miner easier to adapt to an analytical question without exposing the user directly to engine-specific syntax.
- `evaluate()` now reports descriptive quality summaries for the mined patterns, not predictive error.

Common mistakes
- Confusing frequent itemsets with rules; itemsets do not imply direction.
- Setting `supp` too low on rich transactional data and generating an unmanageable number of combinations.
- Forgetting that coercing regular data to transactions may discretize non-categorical columns.
