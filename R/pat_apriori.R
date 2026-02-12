#'@title Apriori rules
#'@description Frequent itemsets and association rules using `arules::apriori`.
#'@param parameter list of parameters passed to `arules::apriori`
#'@param appearance list of item appearance constraints
#'@param control list of control parameters
#'@return returns a `pat_apriori` object
#'@examples
#'data("AdultUCI", package = "arules")
#'trans <- suppressWarnings(methods::as(as.data.frame(AdultUCI), "transactions"))
#'pm <- pat_apriori(parameter = list(
#'  supp = 0.5, conf = 0.9, minlen = 2, maxlen = 10, target = "rules"
#'))
#'pm <- fit(pm, trans)
#'rules <- discover(pm, trans)
#'arules::inspect(rules)
#'@export
pat_apriori <- function(parameter = list(supp = 0.5, conf = 0.9, minlen = 2, maxlen = 10, target = "rules"),
                        appearance = NULL,
                        control = NULL) {
  obj <- pattern_miner()
  obj$parameter <- parameter
  obj$appearance <- appearance
  obj$control <- control
  class(obj) <- append("pat_apriori", class(obj))
  return(obj)
}

#'@importFrom arules apriori
#'@importFrom methods as
#'@exportS3Method discover pat_apriori
discover.pat_apriori <- function(obj, data, ...) {
  if (missing(data)) stop("pat_apriori: data is required.")
  validate_pattern_schema(obj, data)
  if (!inherits(data, "transactions")) {
    data <- methods::as(data, "transactions")
  }
  arules::apriori(
    data,
    parameter = obj$parameter,
    appearance = obj$appearance,
    control = obj$control,
    ...
  )
}
