source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# Classification using Naive Bayes

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
set_example_seed()
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
model <- cla_nb("Species", slevels)
set_example_seed()
model <- fit(model, iris_train)


# Checking fit on training data
train_prediction <- predict(model, iris_train)

# Model evaluation (training)
train_eval <- evaluate(model, iris_train[,"Species"], train_prediction)
print(train_eval$metrics)

# Model test
test_prediction <- predict(model, iris_test)

# Test evaluation
 test_eval <- evaluate(model, iris_test[,"Species"], test_prediction)
print(test_eval$metrics)
