source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris
table(iris$Species)

set_example_seed()
tt <- train_test(sample_stratified("Species"), iris)

table(tt$train$Species)
table(tt$test$Species)

sample <- sample_stratified("Species")
folds <- k_fold(sample, iris, 4)

tbl <- NULL
for (f in folds) {
  tbl <- rbind(tbl, table(f$Species))
}
tbl

slevels <- levels(iris$Species)

model <- cla_majority("Species", slevels)
set_example_seed()
model <- fit(model, tt$train)

prediction <- predict(model, tt$test)
eval <- evaluate(model, tt$test$Species, prediction)
eval$metrics
