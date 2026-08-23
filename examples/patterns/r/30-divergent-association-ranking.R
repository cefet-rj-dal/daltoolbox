source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "arules"))

library(daltoolbox)

data("AdultUCI", package = "arules")

data <- as.data.frame(AdultUCI)
data <- data[, c(
  "workclass",
  "education",
  "marital-status",
  "occupation",
  "income"
)]

rules <- pat_dara_rules(
  data,
  rhs = "income=small",
  supp = 0.05,
  conf = 0.6,
  minlen = 2,
  maxlen = 4,
  remove_redundant = FALSE
)

head(rules[, c("lhs", "rhs", "support", "confidence", "lift", "count")])

none_rules <- pat_filter_rules(rules, pat_rule_filter_none())
nrow(none_rules)

interest_filter <- pat_rule_filter_interest(
  lift_min = 1.2,
  kulc_min = 0.38,
  ir_max = 0.8
)

interest_rules <- pat_filter_rules(rules, interest_filter)
interest_rules[, c("lhs", "lift", "kulc", "ir")]

dara_filter <- pat_rule_filter_dara(
  data,
  rhs_attr = "income",
  rhs_value = "small",
  min_score = 1
)

dara_rules <- pat_filter_rules(rules, dara_filter)
dara_rules[, c("lhs", "dara_score")]

ranking <- attr(dara_rules, "dara_ranking")
ranking[["marital-status"]]

pm <- pat_apriori(
  target = "rules",
  supp = 0.05,
  conf = 0.6,
  minlen = 2,
  maxlen = 4,
  rhs = "income=small",
  rule_filter = dara_filter,
  control = list(verbose = FALSE)
)

pm <- fit(pm, data)
filtered_rules <- discover(pm, data)
length(filtered_rules)
attr(filtered_rules, "dara_score")
