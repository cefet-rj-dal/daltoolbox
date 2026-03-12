# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris
slevels <- levels(iris$Species)

set.seed(1)
sr <- train_test(sample_stratified("Species"), iris)
iris_train <- sr$train
iris_test <- sr$test

baseline <- cla_majority("Species", slevels)
baseline <- fit(baseline, iris_train)

baseline_pred <- predict(baseline, iris_test)
baseline_eval <- evaluate(baseline, adjust_class_label(iris_test$Species), baseline_pred)
baseline_eval$metrics

tree <- cla_dtree("Species", slevels)
tree <- fit(tree, iris_train)

tree_pred <- predict(tree, iris_test)
tree_eval <- evaluate(tree, adjust_class_label(iris_test$Species), tree_pred)
tree_eval$metrics
