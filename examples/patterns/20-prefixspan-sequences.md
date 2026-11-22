About the method
- `pat_prefix_span`: sequential pattern mining (PrefixSpan), where the analytical question is about recurring ordered event structures.

Didactic goal: keep the same pattern-mining line of experiment and change only the data type and pattern family so the reader can compare item co-occurrence with event order.

Environment setup.

``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```

Build the classic Zaki sequence database in long format (`sequenceID`, `eventID`, `item`): one row per item of each timed event.

``` r
seqs <- do.call(rbind, list(
  data.frame(sequenceID = 1, eventID = 10, item = c("C", "D")),
  data.frame(sequenceID = 1, eventID = 15, item = c("A", "B", "C")),
  data.frame(sequenceID = 1, eventID = 20, item = c("A", "B", "F")),
  data.frame(sequenceID = 1, eventID = 25, item = c("A", "C", "D", "F")),
  data.frame(sequenceID = 2, eventID = 15, item = c("A", "B", "F")),
  data.frame(sequenceID = 2, eventID = 20, item = c("E")),
  data.frame(sequenceID = 3, eventID = 10, item = c("A", "B", "F")),
  data.frame(sequenceID = 4, eventID = 10, item = c("D", "G", "H")),
  data.frame(sequenceID = 4, eventID = 20, item = c("B", "F")),
  data.frame(sequenceID = 4, eventID = 25, item = c("A", "G", "H"))
))
head(seqs)
```

```
##   sequenceID eventID item
## 1          1      10    C
## 2          1      10    D
## 3          1      15    A
## 4          1      15    B
## 5          1      15    C
## 6          1      20    A
```

Model configuration.

``` r
pm <- pat_prefix_span(
  support = 0.5,
  maxlen = 3
)
```

Fit and discover patterns.

``` r
pm <- fit(pm, seqs)
seqs_found <- discover(pm, seqs)
nrow(seqs_found)
```

```
## [1] 18
```

Evaluate the discovered patterns.

``` r
eval <- evaluate(pm, seqs_found)
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
head(seqs_found)
```

```
##    sequence support length
## 1     <{A}>    1.00      1
## 2   <{B,F}>    1.00      1
## 3     <{B}>    1.00      1
## 4     <{F}>    1.00      1
## 5 <{A,B,F}>    0.75      1
## 6   <{A,B}>    0.75      1
```

What to observe
- The workflow is unchanged from the other pattern-mining examples.
- The main semantic change is that order now matters, so the discovered structures are sequences rather than itemsets or rules.

References
- Pei, J., Han, J., Mortazavi-Asl, B., Wang, J., Pinto, H., Chen, Q., Dayal, U., and Hsu, M. (2001). PrefixSpan: Mining Sequential Patterns Efficiently by Prefix-Projected Pattern Growth. ICDE.
