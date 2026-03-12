# Stratified sampling dataset

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

iris <- datasets::iris
head(iris)
table(iris$Species)

# Split into train and test

# using stratified sampling
tt <- train_test(sample_stratified("Species"), iris)

# training distribution
print(table(tt$train$Species))

# test distribution
print(table(tt$test$Species))

# Split the dataset into folds

# using stratified sampling
# preparing the dataset into four folds
sample <- sample_stratified("Species")
folds <- k_fold(sample, iris, 4)

# folds distribution
tbl <- NULL
for (f in folds) {
tbl <- rbind(tbl, table(f$Species))
}
print(tbl)
