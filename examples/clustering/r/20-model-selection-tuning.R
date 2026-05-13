source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# Clustering - Tune Kmeans

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox)  

data(iris)

# model training with hyperparameter search
base_model <- cluster_kmeans(k = 2)
base_model$metric <- base_model$clu_utils$metric_silhouette
base_model$selector <- base_model$clu_utils$selector_best
base_model$eval_internal <- list(base_model$clu_utils$metric_silhouette)
base_model$eval_external <- list(base_model$clu_utils$metric_entropy)
model <- clu_tune(base_model, ranges = list(k = 2:10))
set_example_seed()
model <- fit(model, iris[,1:4])
model$k

# run with best parameter
clu <- cluster(model, iris[,1:4])
table(clu)

# internal and external evaluation
eval <- evaluate(model, clu, iris$Species)
eval
