source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

# Discretization and smoothing
# Discretization: transform continuous functions, models, variables, and equations into discrete versions. 

# Smoothing: create an approximating function to capture important patterns, reducing noise and high-frequency variation.

# Defining bin intervals is essential to enable the approximation/discretization.

# General function to evaluate different smoothing techniques

iris <- datasets::iris
head(iris)

# smoothing using regular frequency

obj <- smoothing_freq(n = 2)  
set_example_seed()
obj <- fit(obj, iris$Sepal.Length)
sl.bi <- transform(obj, iris$Sepal.Length)
print(table(sl.bi))
obj$interval

entro <- evaluate(obj, as.factor(names(sl.bi)), iris$Species)
print(entro$entropy)

# Optimizing the number of binnings

opt_obj <- smoothing_freq(n=1:20)
set_example_seed()
obj <- fit(opt_obj, iris$Sepal.Length)
obj$n

set_example_seed()
obj <- fit(obj, iris$Sepal.Length)
sl.bi <- transform(obj, iris$Sepal.Length)
print(table(sl.bi))
