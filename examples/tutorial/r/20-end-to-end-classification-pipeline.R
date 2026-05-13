source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris
slevels <- levels(iris$Species)

set_example_seed()
sr <- train_test(sample_stratified("Species"), iris)
iris_train <- sr$train
iris_test <- sr$test

norm <- minmax()
set_example_seed()
norm <- fit(norm, iris_train)

iris_train_norm <- transform(norm, iris_train)
iris_test_norm <- transform(norm, iris_test)

model <- cla_knn("Species", slevels, k = 5)
set_example_seed()
model <- fit(model, iris_train_norm)

train_prediction <- predict(model, iris_train_norm)
train_eval <- evaluate(model, iris_train_norm$Species, train_prediction)
train_eval$metrics

test_prediction <- predict(model, iris_test_norm)
test_eval <- evaluate(model, iris_test_norm$Species, test_prediction)
test_eval$metrics
