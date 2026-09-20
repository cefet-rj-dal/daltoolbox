source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "rpart"))

library(daltoolbox)
library(rpart)

iris <- datasets::iris
head(iris)

slevels <- levels(iris$Species)

set_example_seed()
sr <- train_test(sample_random(), iris)
iris_train <- sr$train
iris_test <- sr$test

tbl <- rbind(
  table(iris[, "Species"]),
  table(iris_train[, "Species"]),
  table(iris_test[, "Species"])
)
rownames(tbl) <- c("dataset", "training", "test")
tbl

model <- cla_rpart("Species")
model <- fit(model, iris_train)

train_prediction <- predict(model, iris_train)
train_eval <- evaluate(model, iris_train[, "Species"], train_prediction)
train_eval$metrics

test_prediction <- predict(model, iris_test)
test_eval <- evaluate(model, iris_test[, "Species"], test_prediction)
test_eval$metrics
