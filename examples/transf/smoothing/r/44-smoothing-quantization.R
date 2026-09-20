source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

iris <- datasets::iris
head(iris)

obj <- smoothing_quantization(n = 2)
set_example_seed()
obj <- fit(obj, iris$Sepal.Length)
sl.bi <- transform(obj, iris$Sepal.Length)
print(table(sl.bi))
obj$interval

bins <- cut(iris$Sepal.Length, unique(obj$interval.adj), FALSE, include.lowest = TRUE)
entro <- evaluate(obj, bins, iris$Species)
print(entro$entropy)

opt_obj <- smoothing_quantization(n=1:20)
set_example_seed()
obj <- fit(opt_obj, iris$Sepal.Length)
obj$n

set_example_seed()
obj <- fit(obj, iris$Sepal.Length)
sl.bi <- transform(obj, iris$Sepal.Length)
print(table(sl.bi))
