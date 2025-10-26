Sobre a transformação
- `categ_mapping`: converte uma coluna categórica em variáveis binárias (one‑hot). Pode usar codificação com n colunas ou n−1 colunas.


``` r
# Mapeamento categórico
# Um atributo categórico com $n$ valores distintos pode ser mapeado em $n$ atributos binários (one‑hot).

# Também é possível mapear para $n-1$ atributos binários: o caso em que todos os atributos binários são zero representa o último valor categórico (não explícito nas colunas).

# installation 
install.packages("daltoolbox")

# loading DAL
library(daltoolbox)
```

Aplicando mapeamento one-hot para a coluna `Species` em um data frame.

``` r
# conjunto de dados para o exemplo 

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


``` r
# criando o mapeamento categórico

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

Aplicando o mesmo mapeamento para um data frame com única coluna categórica.

``` r
# criando o mapeamento categórico
# Pode ser feito a partir de uma única coluna, mas precisa ser um data frame

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
