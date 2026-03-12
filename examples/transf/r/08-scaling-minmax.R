# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

# Normalization

# Normalization is a technique used to equal strength among variables. 

# It is also important to apply it as an input for some machine learning methods. 

# Example

iris <- datasets::iris  
summary(iris)

# Min-Max 
# Adjust numeric values to 0 (minimum value) - 1 (maximum value).

norm <- minmax()
norm <- fit(norm, iris)
ndata <- transform(norm, iris)
summary(ndata)

ddata <- inverse_transform(norm, ndata)
summary(ddata)
