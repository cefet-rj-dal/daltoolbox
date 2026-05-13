source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
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

tr <- na_removal()
iris.na.omit <- transform(tr, iris.na)
head(iris.na.omit)
nrow(iris.na.omit)
