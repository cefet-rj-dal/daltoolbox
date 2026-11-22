source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

seqs <- do.call(rbind, list(
  data.frame(sequenceID = 1, eventID = 10, item = c("C", "D")),
  data.frame(sequenceID = 1, eventID = 15, item = c("A", "B", "C")),
  data.frame(sequenceID = 1, eventID = 20, item = c("A", "B", "F")),
  data.frame(sequenceID = 1, eventID = 25, item = c("A", "C", "D", "F")),
  data.frame(sequenceID = 2, eventID = 15, item = c("A", "B", "F")),
  data.frame(sequenceID = 2, eventID = 20, item = c("E")),
  data.frame(sequenceID = 3, eventID = 10, item = c("A", "B", "F")),
  data.frame(sequenceID = 4, eventID = 10, item = c("D", "G", "H")),
  data.frame(sequenceID = 4, eventID = 20, item = c("B", "F")),
  data.frame(sequenceID = 4, eventID = 25, item = c("A", "G", "H"))
))
head(seqs)

pm <- pat_prefix_span(
  support = 0.5,
  maxlen = 3
)

pm <- fit(pm, seqs)
seqs_found <- discover(pm, seqs)
nrow(seqs_found)

eval <- evaluate(pm, seqs_found)
eval$metrics

head(seqs_found)
