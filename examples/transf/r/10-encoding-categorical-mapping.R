# Categorical mapping
# A categorical attribute with $n$ distinct values can be mapped into $n$ binary (one‑hot) attributes.

# It is also possible to map into $n-1$ binary attributes: the case where all binary attributes are zero represents the last categorical value (not explicit in columns).

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox)

# dataset for the example 

iris <- datasets::iris
head(iris)

# creating the categorical mapping

cm <- categ_mapping("Species")
iris_cm <- transform(cm, iris)
print(head(iris_cm))

# creating the categorical mapping
# It can be done from a single column, but it must be a data frame

diris <- iris[,"Species", drop=FALSE]
head(diris)

iris_cm <- transform(cm, diris)
print(head(iris_cm))
