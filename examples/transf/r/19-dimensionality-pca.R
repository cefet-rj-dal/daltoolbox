# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

# Dataset for example
iris <- datasets::iris
head(iris)

# creates and fits PCA using the target column as reference
mypca <- dt_pca("Species")
mypca <- fit(mypca, datasets::iris)
iris.pca <- transform(mypca, iris)

print(head(iris.pca))
print(head(mypca$pca.transf))

# Manual definition of the number of components
mypca <- dt_pca("Species", 3)
mypca <- fit(mypca, datasets::iris)
iris.pca <- transform(mypca, iris)
print(head(iris.pca))
print(head(mypca$pca.transf))
