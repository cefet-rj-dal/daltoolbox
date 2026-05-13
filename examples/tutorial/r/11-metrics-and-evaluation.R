source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris
slevels <- levels(iris$Species)

set_example_seed()
sr <- sample_random()
sr <- train_test(sr, iris)

model <- cla_dtree("Species", slevels)
set_example_seed()
model <- fit(model, sr$train)

train_prediction <- predict(model, sr$train)
train_eval <- evaluate(model, sr$train$Species, train_prediction)
train_eval$metrics

test_prediction <- predict(model, sr$test)
test_eval <- evaluate(model, sr$test$Species, test_prediction)
test_eval$metrics

table(predicted = test_eval$prediction, observed = sr$test$Species)
