library(devtools)
load_all()

iris <- datasets::iris
iris <- iris[iris$Species != 'virginica',]
iris$Species <- droplevels(iris$Species)
head(iris)

#extracting the levels for the dataset
slevels <- levels(iris$Species)
slevels

# preparing dataset for random sampling
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

model <- cla_majority("Species", slevels)
model <- fit(model, iris_train)
train_prediction <- predict(model, iris_train)

# Model evaluation
iris_train_predictand <- adjust_class_label(iris_train[,"Species"])
train_eval <- evaluate(model, iris_train_predictand, train_prediction)
print(train_eval$metrics)

# Test
test_prediction <- predict(model, iris_test)

iris_test_predictand <- adjust_class_label(iris_test[,"Species"])
test_eval <- evaluate(model, iris_test_predictand, test_prediction)
print(test_eval$metrics)
