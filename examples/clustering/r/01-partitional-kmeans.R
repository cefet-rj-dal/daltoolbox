source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# Clustering - Kmeans

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox)  

# loading dataset
data(iris)

# clustering method configuration
model <- cluster_kmeans(k=3)
model$eval_internal <- list(
  model$clu_utils$metric_silhouette,
  model$clu_utils$metric_davies_bouldin
)
model$eval_external <- list(
  model$clu_utils$metric_entropy,
  model$clu_utils$metric_purity
)

# model fitting and labeling
set_example_seed()
model <- fit(model, iris[,1:4])
clu <- cluster(model, iris[,1:4])
table(clu)

# evaluate model using internal and external metrics
eval <- evaluate(model, clu, iris$Species)
eval

# Influence of normalization in clustering

iris_minmax <- transform(fit(minmax(), iris), iris)
set_example_seed()
model <- fit(model, iris_minmax[,1:4])
clu <- cluster(model, iris_minmax[,1:4])
table(clu)

# evaluate model using internal and external metrics
eval <- evaluate(model, clu, iris_minmax$Species)
eval
