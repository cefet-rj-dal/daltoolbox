About the transformation
- `aggregation`: summarizes a dataset by groups through named aggregation expressions.

Didactic goal: show that some DAL transformations are not instance-level edits but dataset-level restructuring steps. Aggregation is useful before plotting, reporting, and grouped comparison.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```


``` r
agg <- aggregation(
  "Species",
  mean_sepal = mean(Sepal.Length),
  sd_sepal = sd(Sepal.Length),
  n = n()
)

iris_agg <- transform(agg, datasets::iris)
iris_agg
```

```
##      Species mean_sepal  sd_sepal  n
## 1     setosa      5.006 0.3524897 50
## 2 versicolor      5.936 0.5161711 50
## 3  virginica      6.588 0.6358796 50
```
