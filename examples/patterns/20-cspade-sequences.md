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
```

```
## Error in `patutils()`:
## ! could not find function "patutils"
```

``` r
pm <- pat_cspade(
  support = 0.4,
  maxlen = 3,
  quality_filter = utils$quality_min(support = 0.5),
  control = list(verbose = FALSE)
)
```

```
## Error in `pat_cspade()`:
## ! unused arguments (support = 0.4, maxlen = 3, quality_filter = utils$quality_min(support = 0.5))
```

Fit and discover patterns.

``` r
pm <- fit(pm, x)
```

```
## Error:
## ! object 'pm' not found
```

``` r
seqs <- discover(pm, x)
```

```
## Error:
## ! object 'pm' not found
```

``` r
length(seqs)
```

```
## Error:
## ! object 'seqs' not found
```

Evaluate the discovered patterns.

``` r
eval <- evaluate(pm, seqs)
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
head(as(seqs, "data.frame"))
```

```
## Error:
## ! object 'seqs' not found
```

What to observe
- The workflow is unchanged from the other pattern-mining examples.
- The main semantic change is that order now matters, so the discovered structures are sequences rather than itemsets or rules.
