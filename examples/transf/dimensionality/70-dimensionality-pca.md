About the transformation
- `dt_pca`: Principal Component Analysis (PCA) projects correlated variables onto orthogonal components ordered by explained variance. You can let the tool pick the number of components via an elbow heuristic or set it explicitly.

Method
- Learns principal components via an orthogonal transformation (SVD/eigendecomposition of the covariance matrix), applied to centered/scaled numeric predictors.
- By default, the number of components is chosen via minimum-curvature (elbow) on the cumulative explained variance curve; alternatively set `components` manually.

Didactic goal: focus on what changes in the dataset after each operation. In preprocessing examples, understanding the effect of the transformation is as important as learning the function name.

Environment setup.

``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```

Load data and PCA idea.

``` r
# Dataset for example
iris <- datasets::iris
head(iris)
```

```
##   Sepal.Length Sepal.Width Petal.Length Petal.Width Species
## 1          5.1         3.5          1.4         0.2  setosa
## 2          4.9         3.0          1.4         0.2  setosa
## 3          4.7         3.2          1.3         0.2  setosa
## 4          4.6         3.1          1.5         0.2  setosa
## 5          5.0         3.6          1.4         0.2  setosa
## 6          5.4         3.9          1.7         0.4  setosa
```

PCA
PCA finds a projection capturing the largest possible variance in the data. Below, we fit PCA and transform the dataset.


``` r
# creates and fits PCA using the target column as reference
mypca <- dt_pca("Species")
set_example_seed()
mypca <- fit(mypca, datasets::iris)
iris.pca <- transform(mypca, iris)
```

PCA properties


``` r
print(head(iris.pca))
```

```
##         PC1        PC2 Species
## 1 -2.257141 -0.4784238  setosa
## 2 -2.074013  0.6718827  setosa
## 3 -2.356335  0.3407664  setosa
## 4 -2.291707  0.5953999  setosa
## 5 -2.381863 -0.6446757  setosa
## 6 -2.068701 -1.4842053  setosa
```

``` r
print(head(mypca$pca.transf))
```

```
##                     PC1         PC2
## Sepal.Length  0.5210659 -0.37741762
## Sepal.Width  -0.2693474 -0.92329566
## Petal.Length  0.5804131 -0.02449161
## Petal.Width   0.5648565 -0.06694199
```

Manually set the number of components and repeat the transformation.

``` r
# Manual definition of the number of components
mypca <- dt_pca("Species", 3)
set_example_seed()
mypca <- fit(mypca, datasets::iris)
iris.pca <- transform(mypca, iris)
print(head(iris.pca))
```

```
##         PC1        PC2         PC3 Species
## 1 -2.257141 -0.4784238  0.12727962  setosa
## 2 -2.074013  0.6718827  0.23382552  setosa
## 3 -2.356335  0.3407664 -0.04405390  setosa
## 4 -2.291707  0.5953999 -0.09098530  setosa
## 5 -2.381863 -0.6446757 -0.01568565  setosa
## 6 -2.068701 -1.4842053 -0.02687825  setosa
```

``` r
print(head(mypca$pca.transf))
```

```
##                     PC1         PC2        PC3
## Sepal.Length  0.5210659 -0.37741762  0.7195664
## Sepal.Width  -0.2693474 -0.92329566 -0.2443818
## Petal.Length  0.5804131 -0.02449161 -0.1421264
## Petal.Width   0.5648565 -0.06694199 -0.6342727
```

References
- Pearson, K. (1901). On lines and planes of closest fit to systems of points in space.
- Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components.
