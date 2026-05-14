source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "arules"))

library(daltoolbox)

data("AdultUCI", package = "arules")
trans <- suppressWarnings(methods::as(as.data.frame(AdultUCI), "transactions"))
summary(trans)

utils <- patutils()

pm <- pat_apriori(
  target = "rules",
  supp = 0.2,
  conf = 0.85,
  minlen = 2,
  maxlen = 3,
  rhs = c("native-country=United-States"),
  quality_filter = utils$quality_min(confidence = 0.9, lift = 1.03),
  control = list(verbose = FALSE)
)

pm <- fit(pm, trans)
rules <- suppressWarnings(discover(pm, trans))
length(rules)

eval <- evaluate(pm, rules)
eval$metrics

ord <- order(arules::quality(rules)$lift, arules::quality(rules)$confidence, decreasing = TRUE)
arules::inspect(rules[head(ord, 6)])
