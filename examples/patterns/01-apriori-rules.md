About the method
- `pat_apriori`: association-rule mining with semantic configuration stored in the miner object.

Didactic goal: establish the standard pattern-mining line of experiment used throughout this family with a directional rule example: fixed `rhs`, explicit confidence threshold, and quality filtering by lift.

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

pm <- pat_apriori(
  target = "rules",
  supp = 0.2,
  conf = 0.85,
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
rules <- suppressWarnings(discover(pm, trans))
length(rules)
```

```
## [1] 15
```

Evaluate the discovered patterns.

``` r
eval <- evaluate(pm, rules)
eval$metrics
```

```
##            metric      value      type
## 1   pattern_count 15.0000000 intrinsic
## 2    mean_support  0.2536069 intrinsic
## 3 mean_confidence  0.9312326 intrinsic
## 4       mean_lift  1.0376726 intrinsic
## 5     mean_length  2.9333333 intrinsic
## 6  retained_ratio  0.1181102    filter
```

Inspect a few patterns.

``` r
ord <- order(arules::quality(rules)$lift, arules::quality(rules)$confidence, decreasing = TRUE)
arules::inspect(rules[head(ord, 6)])
```

```
##     lhs                                         rhs                            support   confidence coverage  lift    
## [1] {fnlwgt=[1.23e+04,1.41e+05), race=White} => {native-country=United-States} 0.2746612 0.9513510  0.2887064 1.060090
## [2] {education=HS-grad, race=White}          => {native-country=United-States} 0.2578314 0.9406887  0.2740879 1.048210
## [3] {education-num=[9,10), race=White}       => {native-country=United-States} 0.2578314 0.9406887  0.2740879 1.048210
## [4] {education-num=[10,16], race=White}      => {native-country=United-States} 0.4448016 0.9398252  0.4732812 1.047247
## [5] {fnlwgt=[1.41e+05,2.11e+05), race=White} => {native-country=United-States} 0.2713648 0.9293878  0.2919823 1.035617
## [6] {relationship=Not-in-family, race=White} => {native-country=United-States} 0.2053560 0.9282739  0.2212235 1.034376
##     count
## [1] 13415
## [2] 12593
## [3] 12593
## [4] 21725
## [5] 13254
## [6] 10030
```

What to observe
- The thresholds matter a lot. A didactic example should be configured to return some rules, not an empty result.
- Constraining the `rhs` is useful when you want rules that explain a specific consequent.
- `lift` works as a second filter on top of support and confidence, removing rules that are frequent but not especially informative.
- Later pattern examples will keep this same body and only change the pattern family and configuration.
