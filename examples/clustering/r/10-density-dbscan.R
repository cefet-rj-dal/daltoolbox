source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# Clustering - dbscan

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

# loading dataset
data(iris)

# clustering method configuration
model <- cluster_dbscan(minPts = 3)
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
