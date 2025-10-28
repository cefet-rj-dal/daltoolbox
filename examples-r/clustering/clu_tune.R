# Clustering - Tune Kmeans

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox)  

data(iris)

# ajuste do modelo com busca
model <- clu_tune(cluster_kmeans(k = 0),  ranges = list(k = 1:10))
model <- fit(model, iris[,1:4])
model$k

# execução com melhor parâmetro
clu <- cluster(model, iris[,1:4])
table(clu)

# avaliação externa usando rótulos verdadeiros
eval <- evaluate(model, clu, iris$Species)
eval
