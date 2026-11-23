# Paired classification quality check on the IDENTICAL split exported from Python.
# Reads breast_{train,test}.csv written by classification_breast.py and scores the
# same rows with the R learners, so any Python/R gap is a real adapter difference
# (not split variance).
suppressMessages(devtools::load_all("c:/Git/dal/daltoolbox"))

tr <- read.csv("c:/Git/dal/daltoolbox/develop/parity/breast_train.csv", check.names = FALSE)
te <- read.csv("c:/Git/dal/daltoolbox/develop/parity/breast_test.csv", check.names = FALSE)

slevels <- c("malignant", "benign")
tr$target <- factor(tr$target, levels = slevels)
te$target <- factor(te$target, levels = slevels)

acc <- function(model) {
  model <- fit(model, tr)
  p <- predict(model, te)  # score matrix; evaluate() reorders to slevels + argmax
  ev <- evaluate(model, te$target, p)
  round(as.numeric(ev$metrics$accuracy), 3)
}

models <- list(
  "ClaMajority"  = cla_majority("target", slevels),
  "ClaDtree"     = cla_dtree("target", slevels),
  "ClaKnn(k5)"   = cla_knn("target", slevels, k = 5),
  "ClaNb"        = cla_nb("target", slevels),
  "ClaGlm"       = cla_glm("target", positive = "malignant"),
  "ClaRf(30,5)"  = cla_rf("target", slevels, ntree = 30, mtry = 5),
  "ClaMlp(5,.1)" = cla_mlp("target", slevels, size = 5, decay = 0.1),
  "ClaSvm(def)"  = cla_svm("target", slevels)
)
for (nm in names(models)) {
  val <- tryCatch(acc(models[[nm]]), error = function(e) paste("ERR", conditionMessage(e)))
  cat(sprintf("R  %-14s acc= %s\n", nm, val))
}
cat("DONE\n")
