source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# Pattern mining - eclat itemsets

# installation
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

arules::inspect(itemsets[1:min(6, length(itemsets))])
