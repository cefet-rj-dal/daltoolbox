source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "arules"))

library(daltoolbox)

data("AdultUCI", package = "arules")
trans <- suppressWarnings(methods::as(as.data.frame(AdultUCI), "transactions"))
summary(trans)

utils <- patutils()

pm <- pat_apriori(
  target = "frequent itemsets",
  supp = 0,
  support_strategy = pat_support_threshold("curvature"),
  maxlen = 3,
  include = c("sex=Male", "income=small", "marital-status=Married-civ-spouse", "race=White"),
  quality_filter = utils$quality_min(support = 0.4),
  control = list(verbose = FALSE)
)

pm <- fit(pm, trans)
pm$engine_parameter
itemsets <- discover(pm, trans)
length(itemsets)

eval <- evaluate(pm, itemsets)
eval$metrics

ord <- order(arules::quality(itemsets)$support, decreasing = TRUE)
arules::inspect(itemsets[head(ord, 10)])
