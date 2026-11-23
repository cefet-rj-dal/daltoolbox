# Paired apriori check on the IDENTICAL transactions exported from Python.
# Reads basket.csv (one-hot 0/1), converts to arules transactions, mines the same
# itemsets and rules with the daltoolbox pat_apriori (arules backend).
suppressMessages(devtools::load_all("c:/Git/dal/daltoolbox"))
suppressMessages(library(arules))

basket <- read.csv("c:/Git/dal/daltoolbox/develop/parity/basket.csv", check.names = FALSE)
m <- as.matrix(basket) == 1
storage.mode(m) <- "logical"
trans <- methods::as(m, "transactions")

fmt_set <- function(items) paste0("{", paste(sort(items), collapse = ","), "}")

cat("# frequent itemsets (supp>=0.4)\n")
pm <- pat_apriori(target = "frequent itemsets", supp = 0.4, minlen = 1, maxlen = 3)
pm <- fit(pm, trans)
its <- suppressWarnings(discover(pm, trans))
itdf <- as(its, "data.frame")
itdf <- itdf[order(nchar(gsub("[^,]", "", itdf$items)), itdf$items), ]
for (i in seq_len(nrow(itdf))) {
  items <- gsub("[{}]", "", itdf$items[i])
  cat(sprintf("R   %s  supp=%s\n", fmt_set(strsplit(items, ",")[[1]]), round(itdf$support[i], 3)))
}

cat("# rules (supp>=0.4, conf>=0.6)\n")
pr <- pat_apriori(target = "rules", supp = 0.4, conf = 0.6, minlen = 2)
pr <- fit(pr, trans)
rules <- suppressWarnings(discover(pr, trans))
rdf <- as(rules, "data.frame")
rdf <- rdf[order(-rdf$support, -rdf$confidence), ]
for (i in seq_len(nrow(rdf))) {
  lr <- strsplit(gsub("[{}]", "", rdf$rules[i]), " => ")[[1]]
  lhs <- fmt_set(strsplit(lr[1], ",")[[1]]); rhs <- fmt_set(strsplit(lr[2], ",")[[1]])
  cat(sprintf("R   %s => %s  supp=%s conf=%s lift=%s\n", lhs, rhs,
              round(rdf$support[i], 3), round(rdf$confidence[i], 3), round(rdf$lift[i], 3)))
}
cat("DONE\n")
