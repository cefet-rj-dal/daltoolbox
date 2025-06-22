
``` r
# NA and Outlier analysis

# installation 
install.packages"daltoolbox")

# loading DAL
library(daltoolbox) 
```


``` r
# Outlier removal using boxplot

# The following class uses box-plot definition for outliers.

# An outlier is a value that is below than $Q_1 - 1.5 \cdot IQR$ or higher than $Q_3 + 1.5 \cdot IQR$.
 
# The class remove outliers for numeric attributes. 

# removing outliers of a data frame

# code for outlier removal
out_obj <- outliers_boxplot() # class for outlier analysis
out_obj <- fit(out_obj, iris) # computing boundaries
iris.clean <- transform(out_obj, iris) # returning cleaned dataset

# inspection of cleaned dataset
head(iris.clean)
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

``` r
nrow(iris.clean)
```

```
## [1] 146
```


``` r
# Visualizing the actual outliers

idx <- attr(iris.clean, "idx")
print(table(idx))
```

```
## idx
## FALSE  TRUE 
##   146     4
```

``` r
iris.outliers <- iris[idx,]
head(iris.outliers)
```

```
##    Sepal.Length Sepal.Width Petal.Length Petal.Width    Species
## 16          5.7         4.4          1.5         0.4     setosa
## 33          5.2         4.1          1.5         0.1     setosa
## 34          5.5         4.2          1.4         0.2     setosa
## 61          5.0         2.0          3.5         1.0 versicolor
```

