# Paired quality check on the IDENTICAL split exported from Python.
suppressMessages(devtools::load_all("c:/Git/dal/daltoolbox"))

tr <- read.csv("c:/Git/dal/daltoolbox/develop/parity/diabetes_train.csv", check.names = FALSE)
te <- read.csv("c:/Git/dal/daltoolbox/develop/parity/diabetes_test.csv", check.names = FALSE)

r2 <- function(model) {
  model <- fit(model, tr)
  p <- predict(model, te)
  ev <- evaluate(model, te[, "target"], p)
  round(as.numeric(ev$metrics["R2"]), 3)
}

models <- list(
  "RegLm"            = reg_lm(attribute = "target"),
  "RegDtree"         = reg_dtree("target"),
  "RegKnn"           = reg_knn("target", k = 5),
  "RegRf"            = reg_rf("target", mtry = 7, ntree = 30),
  "RegSvm(eps.2,c40)" = reg_svm("target", epsilon = 0.2, cost = 40.0),
  "RegSvm(def)"      = reg_svm("target"),
  "RegMlp"           = reg_mlp("target", size = 5, decay = 0.54)
)
for (nm in names(models)) {
  val <- tryCatch(r2(models[[nm]]), error = function(e) paste("ERR", conditionMessage(e)))
  cat(sprintf("R  %-18s R2= %s\n", nm, val))
}
cat("DONE\n")
