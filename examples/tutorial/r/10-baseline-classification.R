source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris
slevels <- levels(iris$Species)

set_example_seed()
sr <- train_test(sample_stratified("Species"), iris)
iris_train <- sr$train
iris_test <- sr$test

baseline <- cla_majority("Species", slevels)
set_example_seed()
baseline <- fit(baseline, iris_train)

baseline_pred <- predict(baseline, iris_test)
baseline_eval <- evaluate(baseline, iris_test$Species, baseline_pred)
baseline_eval$metrics

tree <- cla_dtree("Species", slevels)
set_example_seed()
tree <- fit(tree, iris_train)

tree_pred <- predict(tree, iris_test)
tree_eval <- evaluate(tree, iris_test$Species, tree_pred)
tree_eval$metrics
