source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "arulesSequences"))

library(daltoolbox)

x <- arulesSequences::read_baskets(
  con = system.file("misc", "zaki.txt", package = "arulesSequences"),
  info = c("sequenceID", "eventID", "SIZE")
)
x

utils <- patutils()

pm <- pat_cspade(
  support = 0,
  support_strategy = pat_support_threshold("min_count", min_count = 4),
  maxlen = 3,
  quality_filter = utils$quality_min(support = 0.5),
  control = list(verbose = FALSE)
)

pm <- fit(pm, x)
pm$engine_parameter
seqs <- discover(pm, x)
length(seqs)

eval <- evaluate(pm, seqs)
eval$metrics

head(as(seqs, "data.frame"))
