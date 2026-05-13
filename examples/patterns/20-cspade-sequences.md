About the method
- `pat_cspade`: sequential pattern mining, where the analytical question is about recurring ordered event structures.

Didactic goal: keep the same pattern-mining line of experiment and change only the data type and pattern family so the reader can compare item co-occurrence with event order.

Environment setup.

``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "arulesSequences"))

library(daltoolbox)
```

Load sequence transactions.

``` r
x <- arulesSequences::read_baskets(
  con = system.file("misc", "zaki.txt", package = "arulesSequences"),
  info = c("sequenceID", "eventID", "SIZE")
)
x
```

```
## transactions in sparse format with
##  10 transactions (rows) and
##  8 items (columns)
```

Model configuration.

``` r
utils <- patutils()

pm <- pat_cspade(
  support = 0.4,
  maxlen = 3,
  quality_filter = utils$quality_min(support = 0.5),
  control = list(verbose = FALSE)
)
```

Fit and discover patterns.

``` r
pm <- fit(pm, x)
seqs <- discover(pm, x)
length(seqs)
```

```
## [1] 18
```

Evaluate the discovered patterns.

``` r
eval <- evaluate(pm, seqs)
eval$metrics
```

```
##           metric      value      type
## 1  pattern_count 18.0000000 intrinsic
## 2   mean_support  0.6527778 intrinsic
## 3    mean_length  1.7222222 intrinsic
## 4 retained_ratio  1.0000000    filter
```

Inspect a few patterns.

``` r
head(as(seqs, "data.frame"))
```

```
##   sequence support
## 1    <{A}>    1.00
## 2    <{B}>    1.00
## 3    <{D}>    0.50
## 4    <{F}>    1.00
## 5  <{A,F}>    0.75
## 6  <{B,F}>    1.00
```

What to observe
- The workflow is unchanged from the other pattern-mining examples.
- The main semantic change is that order now matters, so the discovered structures are sequences rather than itemsets or rules.
