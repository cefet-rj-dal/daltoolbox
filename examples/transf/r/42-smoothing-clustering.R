source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

iris <- datasets::iris
cluster_data <- iris[, c("Sepal.Length", "Species")]
head(cluster_data)

# smoothing using class-aware clustering
obj <- smoothing_cluster("Species", n = 3)
set_example_seed()
obj <- fit(obj, cluster_data)
sl.bi <- transform(obj, iris$Sepal.Length)
print(table(sl.bi))
obj$interval

bins <- cut(iris$Sepal.Length, unique(obj$interval.adj), FALSE, include.lowest = TRUE)
entro <- evaluate(obj, bins, iris$Species)
print(entro$entropy)

opt_obj <- smoothing_cluster("Species", n=1:8)
set_example_seed()
obj <- fit(opt_obj, cluster_data)
obj$n

set_example_seed()
obj <- fit(obj, cluster_data)
sl.bi <- transform(obj, iris$Sepal.Length)
print(table(sl.bi))
