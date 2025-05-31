# Categorical mapping
A categorical attribute with $n$ distinct values is mapped into $n$ binary attributes. 

It is also possible to map into $n-1$ binary values, where the scenario where all binary attributes are equal to zero corresponds to the last categorical value not indicated in the attributes.  


``` r
# DAL ToolBox
# version 1.2.707



# loading DAL
library(daltoolbox)
```

# dataset for example 


``` r
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

# creating categorical mapping


``` r
cm <- categ_mapping("Species")
iris_cm <- transform(cm, iris)
print(head(iris_cm))
```

```
##   Speciessetosa Speciesversicolor Speciesvirginica
## 1             1                 0                0
## 2             1                 0                0
## 3             1                 0                0
## 4             1                 0                0
## 5             1                 0                0
## 6             1                 0                0
```

# creating categorical mapping
Can be made from a single column, but needs to be a data frame


``` r
diris <- iris[,"Species", drop=FALSE]
head(diris)
```

```
##   Species
## 1  setosa
## 2  setosa
## 3  setosa
## 4  setosa
## 5  setosa
## 6  setosa
```


``` r
iris_cm <- transform(cm, diris)
print(head(iris_cm))
```

```
##   Speciessetosa Speciesversicolor Speciesvirginica
## 1             1                 0                0
## 2             1                 0                0
## 3             1                 0                0
## 4             1                 0                0
## 5             1                 0                0
## 6             1                 0                0
```

