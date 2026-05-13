source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "arules"))

library(daltoolbox)

data("AdultUCI", package = "arules")
adult_df <- as.data.frame(AdultUCI)
head(adult_df)

utils <- patutils()

pm <- pat_eclat(
  supp = 0.5,
  maxlen = 3,
  include = c("sex=Male", "income=small", "marital-status=Married-civ-spouse"),
  quality_filter = utils$quality_min(support = 0.55)
)

pm <- fit(pm, adult_df)
itemsets <- discover(pm, adult_df)
length(itemsets)

eval <- evaluate(pm, itemsets)
eval$metrics

if (length(itemsets) == 0) {
  cat("No itemsets remained after the quality filter. Lower the thresholds if you want to inspect some candidates.\n")
} else {
  arules::inspect(itemsets[seq_len(min(6, length(itemsets)))])
}
