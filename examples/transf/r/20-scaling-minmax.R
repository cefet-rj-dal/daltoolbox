source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
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
set_example_seed()
norm <- fit(norm, iris)
ndata <- transform(norm, iris)
summary(ndata)

ddata <- inverse_transform(norm, ndata)
summary(ddata)
