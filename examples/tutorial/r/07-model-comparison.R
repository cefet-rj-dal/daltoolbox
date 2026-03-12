# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris
slevels <- levels(iris$Species)

set.seed(1)
sr <- train_test(sample_stratified("Species"), iris)
iris_train <- sr$train
iris_test <- sr$test

models <- list(
  majority = cla_majority("Species", slevels),
  tree = cla_dtree("Species", slevels),
  knn = cla_knn("Species", slevels, k = 5),
  rf = cla_rf("Species", slevels, mtry = 2, ntree = 10)
)

results <- lapply(models, function(model) {
  fitted <- fit(model, iris_train)
  pred <- predict(fitted, iris_test)
  evaluate(fitted, adjust_class_label(iris_test$Species), pred)$metrics
})

results
