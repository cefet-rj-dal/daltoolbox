About the transformation
- `feature_generation`: creates new attributes from named expressions over existing columns.

Didactic goal: reinforce that preprocessing is not only about cleaning or scaling. Feature generation changes the representation itself and can make patterns easier for downstream learners to capture.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```


``` r
gen <- feature_generation(
  Sepal.Area = Sepal.Length * Sepal.Width,
  Petal.Area = Petal.Length * Petal.Width,
  Sepal.Ratio = Sepal.Length / Sepal.Width
)

iris_feat <- transform(gen, datasets::iris)
head(iris_feat)
```

```
##   Sepal.Length Sepal.Width Petal.Length Petal.Width Species Sepal.Area Petal.Area Sepal.Ratio
## 1          5.1         3.5          1.4         0.2  setosa      17.85       0.28    1.457143
## 2          4.9         3.0          1.4         0.2  setosa      14.70       0.28    1.633333
## 3          4.7         3.2          1.3         0.2  setosa      15.04       0.26    1.468750
## 4          4.6         3.1          1.5         0.2  setosa      14.26       0.30    1.483871
## 5          5.0         3.6          1.4         0.2  setosa      18.00       0.28    1.388889
## 6          5.4         3.9          1.7         0.4  setosa      21.06       0.68    1.384615
```

What to observe
- Generated attributes become regular columns in the transformed dataset.
- This is often one of the most impactful preprocessing steps in practice because it injects domain-informed structure.
