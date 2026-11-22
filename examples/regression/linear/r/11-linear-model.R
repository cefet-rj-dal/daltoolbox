source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

data(Diabetes)
head(Diabetes)

set_example_seed()
sr <- train_test(sample_random(), Diabetes)
diabetes_train <- sr$train
diabetes_test <- sr$test

model <- reg_lm(attribute = "target")
model <- fit(model, diabetes_train)

train_prediction <- predict(model, diabetes_train)
train_eval <- evaluate(model, diabetes_train[, "target"], train_prediction)
train_eval$metrics

test_prediction <- predict(model, diabetes_test)
test_eval <- evaluate(model, diabetes_test[, "target"], test_prediction)
test_eval$metrics
