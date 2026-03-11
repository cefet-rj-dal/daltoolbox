# installation
# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris
iris$Species <- factor(iris$Species)

fs <- feature_selection_relief("Species", top = 2, m = 50, seed = 1)
fs <- fit(fs, iris)

print(fs$selected)
print(fs$ranking)

iris_fs <- transform(fs, iris)
head(iris_fs)
