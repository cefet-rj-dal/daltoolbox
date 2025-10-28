# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

# Dataset for example
iris <- datasets::iris
head(iris)

# cria e ajusta PCA usando a coluna alvo para referência
mypca <- dt_pca("Species")
mypca <- fit(mypca, datasets::iris)
iris.pca <- transform(mypca, iris)

print(head(iris.pca))
print(head(mypca$pca.transf))

# Definição manual do número de componentes
mypca <- dt_pca("Species", 3)
mypca <- fit(mypca, datasets::iris)
iris.pca <- transform(mypca, iris)
print(head(iris.pca))
print(head(mypca$pca.transf))
