source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

iris <- datasets::iris
head(iris)

# smoothing using clustering
obj <- smoothing_cluster(n = 2)  
set_example_seed()
obj <- fit(obj, iris$Sepal.Length)
sl.bi <- transform(obj, iris$Sepal.Length)
print(table(sl.bi))
obj$interval

entro <- evaluate(obj, as.factor(names(sl.bi)), iris$Species)
print(entro$entropy)

opt_obj <- smoothing_cluster(n=1:20)
set_example_seed()
obj <- fit(opt_obj, iris$Sepal.Length)
obj$n

set_example_seed()
obj <- fit(obj, iris$Sepal.Length)
sl.bi <- transform(obj, iris$Sepal.Length)
print(table(sl.bi))
