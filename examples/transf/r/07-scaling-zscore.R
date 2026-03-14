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

# Z-Score

# Adjust values to 0 (mean), 1 (variance).

norm <- zscore()
norm <- fit(norm, iris)
ndata <- transform(norm, iris)
summary(ndata)

ddata <- inverse_transform(norm, ndata)
summary(ddata)

norm <- zscore(nmean=0.5, nsd=0.5/2.698)
norm <- fit(norm, iris)
ndata <- transform(norm, iris)
summary(ndata)

ddata <- inverse_transform(norm, ndata)
summary(ddata)
