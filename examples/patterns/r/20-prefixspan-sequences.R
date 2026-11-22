source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

# Sequences as a long data frame: sequenceID / eventID / item
# (the classic Zaki example: four sequences of timed itemsets).
seqs <- data.frame(
  sequenceID = c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 2, 2, 2, 2,
                 3, 3, 3,
                 4, 4, 4, 4, 4, 4, 4, 4),
  eventID = c(10, 10, 15, 15, 15, 20, 20, 20, 25, 25, 25, 25, 25,
              15, 15, 15, 20,
              10, 10, 10,
              10, 10, 10, 20, 20, 25, 25, 25),
  item = c("C", "D", "A", "B", "C", "A", "B", "F", "A", "C", "D", "F", "F",
           "A", "B", "F", "E",
           "A", "B", "F",
           "D", "G", "H", "B", "F", "A", "G", "H"),
  stringsAsFactors = FALSE
)

pm <- pat_prefix_span(
  support = 0.5,
  minlen = 1,
  maxlen = 3
)

pm <- fit(pm, seqs)
pm$support_resolved

seqs_found <- discover(pm, seqs)
nrow(seqs_found)
head(seqs_found, 10)

# Evaluate the discovered sequences
eval <- evaluate(pm, seqs_found)
eval$metrics

# Time-gap constraint: only patterns whose matched events are at most 5 apart
pm_gap <- pat_prefix_span(support = 0.5, maxgap = 5)
pm_gap <- fit(pm_gap, seqs)
discover(pm_gap, seqs)$sequence
