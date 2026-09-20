source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "glmnet"))

library(daltoolbox)
library(glmnet)

iris_bin <- datasets::iris
iris_bin$IsVersicolor <- factor(ifelse(
  iris_bin$Species == "versicolor",
  "versicolor",
  "not_versicolor"
))
head(iris_bin)

slevels <- levels(iris_bin$IsVersicolor)

set_example_seed()
sr <- train_test(sample_stratified("IsVersicolor"), iris_bin)
iris_train <- sr$train
iris_test <- sr$test

tbl <- rbind(
  table(iris_bin[, "IsVersicolor"]),
  table(iris_train[, "IsVersicolor"]),
  table(iris_test[, "IsVersicolor"])
)
rownames(tbl) <- c("dataset", "training", "test")
tbl

model <- cla_glmnet("IsVersicolor", lambda = "lambda.1se")
model <- fit(model, iris_train)

train_prediction <- predict(model, iris_train)
train_eval <- evaluate(model, iris_train[, "IsVersicolor"], train_prediction)
train_eval$metrics

test_prediction <- predict(model, iris_test)
test_eval <- evaluate(model, iris_test[, "IsVersicolor"], test_prediction)
test_eval$metrics
