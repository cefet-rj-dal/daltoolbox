# Paired clustering quality check on the IDENTICAL rows exported from Python.
# Reads clu_iris.csv written by clustering_iris.py and scores external agreement of
# the partition with the reference labels (Species): entropy (lower = better) and
# purity (higher = better). Any Python/R gap is a real adapter/algorithm difference.
suppressMessages(devtools::load_all("c:/Git/dal/daltoolbox"))

iris <- read.csv("c:/Git/dal/daltoolbox/develop/parity/clu_iris.csv", check.names = FALSE)
x <- iris[, setdiff(names(iris), "Species")]
ref <- as.factor(iris$Species)
utils <- cluutils()

score <- function(model) {
  model <- fit(model, x)
  clu <- cluster(model, x)
  ent <- utils$metric_entropy(clu, ref)$value
  pur <- utils$metric_purity(clu, ref)$value
  sprintf("entropy= %.3f  purity= %.3f", ent, pur)
}

models <- list(
  "Kmeans(k3)" = cluster_kmeans(k = 3),
  "Pam(k3)"    = cluster_pam(k = 3),
  "Cmeans(c3)" = cluster_cmeans(centers = 3),
  "Gmm(g3)"    = cluster_gmm(G = 3),
  "Hclust(k3)" = cluster_hclust(k = 3),
  "Dbscan"     = cluster_dbscan(minPts = 3)
)
for (nm in names(models)) {
  val <- tryCatch(score(models[[nm]]), error = function(e) paste("ERR", conditionMessage(e)))
  cat(sprintf("R  %-12s %s\n", nm, val))
}
cat("DONE\n")
