# Paired transformation check on the IDENTICAL input exported from Python.
# Writes out_r_<name>.csv for each deterministic transform; transforms_compare.py
# diffs the pairs (out_py_* vs out_r_*) up to a tiny tolerance.
suppressMessages(devtools::load_all("c:/Git/dal/daltoolbox"))

iris <- read.csv("c:/Git/dal/daltoolbox/develop/parity/iris_full.csv", check.names = FALSE)
iris$Species <- as.factor(iris$Species)
iris_na <- read.csv("c:/Git/dal/daltoolbox/develop/parity/iris_na.csv", check.names = FALSE)
iris_na$Species <- as.factor(iris_na$Species)

dump <- function(name, df) {
  write.csv(df, sprintf("c:/Git/dal/daltoolbox/develop/parity/out_r_%s.csv", name), row.names = FALSE)
  cat(sprintf("R   %-22s shape=(%d, %d)\n", name, nrow(df), ncol(df)))
}

app <- function(obj, data) transform(fit(obj, data), data)

dump("minmax", app(minmax(), iris))
dump("zscore", app(zscore(), iris))

cm <- app(categ_mapping("Species"), iris)
dump("categ_mapping", cm)

pca <- app(dt_pca("Species", components = 2), iris)
numcols <- sapply(pca, is.numeric)
pca_abs <- pca
pca_abs[numcols] <- abs(pca[numcols])  # PCA sign arbitrary
dump("dt_pca_abs", pca_abs)

dump("na_removal", app(na_removal(), iris_na))
dump("imputation_median", app(imputation_simple(method = "median"), iris_na))
dump("outliers_boxplot", app(outliers_boxplot(), iris))
dump("outliers_gaussian", app(outliers_gaussian(), iris))
cat("DONE\n")
