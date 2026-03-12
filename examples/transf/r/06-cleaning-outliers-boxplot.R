# NA and Outlier analysis

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

# Outlier removal using boxplot rule

# The class uses the boxplot rule to define outliers.

# An outlier is a value smaller than $Q_1 - 1.5\cdot IQR$ or larger than $Q_3 + 1.5\cdot IQR$.
 
# The class removes outliers in numeric attributes.

# Removing outliers from a data frame

# Example outlier removal code
out_obj <- outliers_boxplot() # outlier analysis class
out_obj <- fit(out_obj, iris) # computes limits via quartiles and IQR
iris.clean <- transform(out_obj, iris) # returns cleaned dataset

# inspection of cleaned dataset
head(iris.clean)
nrow(iris.clean)

# Visualizing the actual outliers

idx <- attr(iris.clean, "idx")
print(table(idx))
iris.outliers <- iris[idx,]
head(iris.outliers)
