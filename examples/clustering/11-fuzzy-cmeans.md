About the method
- `cluster_cmeans`: fuzzy c-means clustering, which keeps membership degrees instead of only hard assignments.

Didactic goal: keep the same clustering line of experiment and change only the notion of assignment from hard clusters to soft memberships.

Environment setup.

``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```

Load data and separate predictors from the reference labels used only for interpretation.

``` r
iris <- datasets::iris
x <- iris[, 1:4]
ref <- iris$Species
head(x)
```

```
##   Sepal.Length Sepal.Width Petal.Length Petal.Width
## 1          5.1         3.5          1.4         0.2
## 2          4.9         3.0          1.4         0.2
## 3          4.7         3.2          1.3         0.2
## 4          4.6         3.1          1.5         0.2
## 5          5.0         3.6          1.4         0.2
## 6          5.4         3.9          1.7         0.4
```

Model configuration.

``` r
model <- cluster_cmeans(centers = 3, m = 2)
```

Fit the model and obtain cluster labels.

``` r
set_example_seed()
model <- fit(model, x)
clu <- cluster(model, x)
table(clu)
```

```
## clu
##  1  2  3 
## 50 60 40
```

Evaluate the partition.

``` r
eval <- evaluate(model, clu, ref)
eval
```

```
## $clusters_entropy
## # A tibble: 3 × 4
##   x        ce   qtd   ceg
##   <fct> <dbl> <int> <dbl>
## 1 1     0        50 0    
## 2 2     0.754    60 0.302
## 3 3     0.384    40 0.102
## 
## $clustering_entropy
## [1] 0.4040967
## 
## $data_entropy
## [1] 1.584963
## 
## $metrics
##                metric     value     goal     type
## 1          silhouette 0.5495175 maximize internal
## 2         withinerror 0.4033714 minimize    model
## 3             entropy 0.4040967 minimize external
## 4              purity 0.8933333 maximize external
## 5 adjusted_rand_index 0.7294203 maximize external
```

Inspect the membership matrix attached to the result.

``` r
head(attr(clu, "membership"))
```

```
##              1           2           3
## [1,] 0.9966236 0.002304388 0.001072020
## [2,] 0.9758508 0.016650843 0.007498391
## [3,] 0.9798248 0.013760371 0.006414865
## [4,] 0.9674251 0.022466803 0.010108106
## [5,] 0.9944703 0.003761757 0.001767929
## [6,] 0.9345703 0.044809020 0.020620672
```

References
- Bezdek, J. C. (1981). Pattern Recognition with Fuzzy Objective Function Algorithms.
