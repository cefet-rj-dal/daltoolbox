# installation
# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris
iris$Species <- factor(iris$Species)

fs <- feature_selection_info_gain("Species", top = 2)
fs <- fit(fs, iris)

print(fs$selected)
print(fs$ranking)

iris_fs <- transform(fs, iris)
head(iris_fs)
