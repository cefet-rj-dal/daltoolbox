## Tutorial 03 - Data Quality and Cleaning

In real projects, modeling rarely starts on a perfectly clean table. Missing values, unusual observations, and categorical attributes often need attention before the first learner is trained. This tutorial introduces that preparation mindset.

The purpose is not to exhaust every cleaning strategy, but to show that `daltoolbox` treats these steps as explicit workflow operations rather than ad hoc edits.


``` r
# install.packages("daltoolbox")

library(daltoolbox)
```

Create a small didactic dataset with missing values. A synthetic example is useful here because the effect of cleaning becomes easier to see.

``` r
small <- iris[1:12, ]
small$Sepal.Length[c(3, 8)] <- NA
small$Sepal.Width[5] <- NA
small
```

```
##    Sepal.Length Sepal.Width Petal.Length Petal.Width Species
## 1           5.1         3.5          1.4         0.2  setosa
## 2           4.9         3.0          1.4         0.2  setosa
## 3            NA         3.2          1.3         0.2  setosa
## 4           4.6         3.1          1.5         0.2  setosa
## 5           5.0          NA          1.4         0.2  setosa
## 6           5.4         3.9          1.7         0.4  setosa
## 7           4.6         3.4          1.4         0.3  setosa
## 8            NA         3.4          1.5         0.2  setosa
## 9           4.4         2.9          1.4         0.2  setosa
## 10          4.9         3.1          1.5         0.1  setosa
## 11          5.4         3.7          1.5         0.2  setosa
## 12          4.8         3.4          1.6         0.2  setosa
```

A first simple option is to remove incomplete rows. This is not always the best strategy in practice, but it is an important baseline cleaning operation.

``` r
small_complete <- na.omit(small)
small_complete
```

```
##    Sepal.Length Sepal.Width Petal.Length Petal.Width Species
## 1           5.1         3.5          1.4         0.2  setosa
## 2           4.9         3.0          1.4         0.2  setosa
## 4           4.6         3.1          1.5         0.2  setosa
## 6           5.4         3.9          1.7         0.4  setosa
## 7           4.6         3.4          1.4         0.3  setosa
## 9           4.4         2.9          1.4         0.2  setosa
## 10          4.9         3.1          1.5         0.1  setosa
## 11          5.4         3.7          1.5         0.2  setosa
## 12          4.8         3.4          1.6         0.2  setosa
```

Now inspect a basic outlier strategy on a numeric attribute. The point is to show that the analyst can explicitly choose how to detect suspicious observations before modeling.

``` r
out <- outliers_boxplot(features = c("Sepal.Length"))
```

```
## Error in `outliers_boxplot()`:
## ! unused argument (features = c("Sepal.Length"))
```

``` r
out <- fit(out, iris)
```

```
## Error:
## ! object 'out' not found
```

``` r
outliers_found <- transform(out, iris)
```

```
## Error:
## ! object 'out' not found
```

``` r
head(outliers_found)
```

```
## Error:
## ! object 'outliers_found' not found
```

Categorical attributes may also need transformation before some learners can use them effectively. The next block shows one-hot style mapping on a toy categorical column.

``` r
cat_data <- data.frame(
  color = factor(c("red", "blue", "green", "red")),
  value = c(10, 20, 15, 12)
)

mapper <- categ_mapping(features = "color")
```

```
## Error in `categ_mapping()`:
## ! unused argument (features = "color")
```

``` r
mapper <- fit(mapper, cat_data)
```

```
## Error:
## ! object 'mapper' not found
```

``` r
cat_encoded <- transform(mapper, cat_data)
```

```
## Error:
## ! object 'mapper' not found
```

``` r
cat_encoded
```

```
## Error:
## ! object 'cat_encoded' not found
```

From a teaching perspective, this tutorial matters because many beginners jump directly to modeling. In practice, data quality decisions often shape the experiment before the learner ever sees the data.
