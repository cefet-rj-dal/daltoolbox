# NA and Outlier analysis

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

# Outlier removal using Gaussian rule (3σ)
# An outlier is a value smaller than $\overline{x} - 3\,\sigma_x$ or larger than $\overline{x} + 3\,\sigma_x$.

# The class removes outliers in numeric attributes.

# Removing outliers from a data frame

# Example outlier removal code
out_obj <- outliers_gaussian() # outlier analysis class
out_obj <- fit(out_obj, iris)  # computes limits based on mean and std dev
iris.clean <- transform(out_obj, iris) # returns cleaned dataset

# inspection of cleaned dataset
head(iris.clean)
nrow(iris.clean)

# Visualizing the actual outliers

idx <- attr(iris.clean, "idx")
print(table(idx))
iris.outliers <- iris[idx,]
head(iris.outliers)
