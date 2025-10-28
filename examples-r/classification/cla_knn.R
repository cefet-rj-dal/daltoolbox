# Classification using KNN

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

iris <- datasets::iris
head(iris)

# extracting the levels for the dataset
slevels <- levels(iris$Species)
slevels

# Building train and test samples via random sampling
set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)
iris_train <- sr$train
iris_test <- sr$test


tbl <- rbind(table(iris[,"Species"]), 
             table(iris_train[,"Species"]), 
             table(iris_test[,"Species"]))
rownames(tbl) <- c("dataset", "training", "test")
head(tbl)

# Model training
model <- cla_knn("Species", slevels, k=1) # k=1 for nearest neighbor
model <- fit(model, iris_train)


# Checking fit on training data
train_prediction <- predict(model, iris_train)

# Model evaluation (training)
iris_train_predictand <- adjust_class_label(iris_train[,"Species"])
train_eval <- evaluate(model, iris_train_predictand, train_prediction)
print(train_eval$metrics)

# Model test
test_prediction <- predict(model, iris_test)

iris_test_predictand <- adjust_class_label(iris_test[,"Species"])

# Test evaluation
 test_eval <- evaluate(model, iris_test_predictand, test_prediction)
print(test_eval$metrics)
