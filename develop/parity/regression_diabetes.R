# Paired quality check: regression models on the Diabetes dataset (R side).
# Compares test R2 across model families to check that ML models are not beaten
# by the linear model due to weak adapter defaults.
suppressMessages(devtools::load_all("c:/Git/dal/daltoolbox"))

data(Diabetes)
set.seed(1)
sr <- train_test(sample_random(), Diabetes)
tr <- sr$train
te <- sr$test

r2 <- function(model) {
  model <- fit(model, tr)
  p <- predict(model, te)
  ev <- evaluate(model, te[, "target"], p)
  as.numeric(ev$metrics["R2"])
}

models <- list(
  "reg_lm (linear)" = reg_lm("target"),
  "reg_dtree"       = reg_dtree("target"),
  "reg_knn"         = reg_knn("target", k = 5),
  "reg_rf"          = reg_rf("target", mtry = 7, ntree = 30),
  "reg_svm(eps=0.2,cost=40)" = reg_svm("target", epsilon = 0.2, cost = 40.0),
  "reg_svm(default)" = reg_svm("target"),
  "reg_mlp"         = reg_mlp("target", size = 5, decay = 0.54)
)

for (nm in names(models)) {
  val <- tryCatch(round(r2(models[[nm]]), 3), error = function(e) paste("ERR", conditionMessage(e)))
  cat(sprintf("%-28s R2= %s\n", nm, val))
}
cat("PARITY_R_DONE\n")
