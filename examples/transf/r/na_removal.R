# NA and Outlier analysis

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

# NA removal

iris <- datasets::iris
head(iris)
nrow(iris)

# introducing an NA to remove

iris.na <- iris
iris.na$Sepal.Length[2] <- NA
head(iris.na)
nrow(iris.na)

# removing rows with NA

iris.na.omit <- na.omit(iris.na)
head(iris.na.omit)
nrow(iris.na.omit)
