source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# Pattern mining - apriori rules

# installation
# install.packages(c("daltoolbox", "arules"))

library(daltoolbox)

data("AdultUCI", package = "arules")
trans <- suppressWarnings(methods::as(as.data.frame(AdultUCI), "transactions"))
summary(trans)

utils <- patutils()

pm <- pat_apriori(
  target = "rules",
  supp = 0.5,
  conf = 0.9,
  minlen = 2,
  maxlen = 4,
  rhs = c("income=small"),
  quality_filter = utils$quality_min(confidence = 0.95, lift = 1.05)
)

pm <- fit(pm, trans)
rules <- discover(pm, trans)
length(rules)

eval <- evaluate(pm, rules)
eval$metrics

arules::inspect(rules[1:min(6, length(rules))])
