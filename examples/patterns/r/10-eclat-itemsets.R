source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "arules"))

library(daltoolbox)

data("AdultUCI", package = "arules")
trans <- suppressWarnings(methods::as(as.data.frame(AdultUCI), "transactions"))
summary(trans)

utils <- patutils()

pm <- pat_eclat(
  supp = 0.2,
  maxlen = 3,
  include = c("sex=Male", "income=small", "marital-status=Married-civ-spouse", "race=White"),
  exclude = c("income=small"),
  quality_filter = utils$quality_min(support = 0.4),
  control = list(verbose = FALSE)
)

pm <- fit(pm, trans)
itemsets <- discover(pm, trans)
length(itemsets)

eval <- evaluate(pm, itemsets)
eval$metrics

ord <- order(arules::quality(itemsets)$support, decreasing = TRUE)
arules::inspect(itemsets[ord])
