About the transformation
- `sample_cluster`: samples whole groups defined by a categorical attribute.

Didactic goal: show a sampling strategy where the unit of selection is the group, not the individual row. This matters when preserving within-group structure is more important than preserving the global distribution exactly.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```


``` r
sc <- sample_cluster("Species", n_clusters = 2)
set_example_seed()
iris_sc <- transform(sc, datasets::iris)
table(iris_sc$Species)
```

```
## 
##     setosa versicolor  virginica 
##         50          0         50
```

What to observe
- Entire species groups are kept or removed together in this toy example.
- In practice, this kind of sampling is useful when rows belong to natural blocks that should not be split arbitrarily.
