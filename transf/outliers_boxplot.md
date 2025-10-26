
``` r
# NA and Outlier analysis

Sobre a transformação
- `outliers_boxplot`: identifica outliers por regra do boxplot (Q1 − 1.5·IQR, Q3 + 1.5·IQR) e pode removê-los de atributos numéricos.

# installation 
install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```

```
## Error in parse(text = input): <text>:3:7: unexpected symbol
## 2: 
## 3: Sobre a
##          ^
```

Removendo outliers via boxplot e inspecionando o resultado.

``` r
# Remoção de outliers usando boxplot

# A classe utiliza a regra do boxplot para definir outliers.

# Um outlier é um valor menor que $Q_1 - 1{,}5\cdot IQR$ ou maior que $Q_3 + 1{,}5\cdot IQR$.
 
# A classe remove outliers em atributos numéricos.

# removendo outliers de um data frame

# código para remoção de outliers
out_obj <- outliers_boxplot() # classe para análise de outliers
out_obj <- fit(out_obj, iris) # computa limites por quartis e IQR
iris.clean <- transform(out_obj, iris) # retorna conjunto de dados limpo

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

Visualizando quais linhas foram marcadas como outliers.

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
