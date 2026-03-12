# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris
table(iris$Species)

set.seed(1)
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
model <- fit(model, tt$train)

prediction <- predict(model, tt$test)
eval <- evaluate(model, adjust_class_label(tt$test$Species), prediction)
eval$metrics
