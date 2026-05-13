About the method
- `pat_cspade`: sequential pattern mining. The central analytical question is no longer “which items co-occur?” but “which ordered event structures recur often enough to matter?”

Didactic goal: read this example as a sequence-discovery workflow. The most important lesson is that order-sensitive mining can still follow the same Experiment Line: configure the object, fit it, discover patterns, evaluate the result.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# Pattern mining - cspade sequences

# installation
# install.packages(c("daltoolbox", "arulesSequences"))

library(daltoolbox)
```

Load sequence transactions from the example dataset distributed with `arulesSequences`.

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

Configure the miner with support and sequence-size restrictions.

``` r
utils <- patutils()

pm <- pat_cspade(
  support = 0.4,
  maxlen = 3,
  quality_filter = utils$quality_min(support = 0.5),
  control = list(verbose = FALSE)
)
```

Fit and discover sequential patterns.

``` r
pm <- fit(pm, x)
seqs <- discover(pm, x)
length(seqs)
```

```
## [1] 18
```

Evaluate the resulting sequence set.

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
- `support` remains central, but now the patterns are ordered structures rather than unordered itemsets.
- `maxlen` controls sequence complexity and can be decisive for keeping the result interpretable.
- `evaluate()` summarizes the discovered sequence family without pretending that this is a supervised learning problem.

Common mistakes
- Reading sequential patterns as if they were ordinary association rules.
- Using very loose support thresholds and letting the number of sequences explode.
- Ignoring sequence length when interpreting the result.
