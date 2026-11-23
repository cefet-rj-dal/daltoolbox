# Paired feature-selection check on the IDENTICAL rows exported from Python.
# For each selector the selected SET = names(transform(fit(fs, data), data)) minus the
# target. Compares against the Python `.selected` sets (featsel.py) on the same rows.
suppressMessages(devtools::load_all("c:/Git/dal/daltoolbox"))

diabetes <- read.csv("c:/Git/dal/daltoolbox/develop/parity/diabetes_full.csv", check.names = FALSE)
breast <- read.csv("c:/Git/dal/daltoolbox/develop/parity/breast_full.csv", check.names = FALSE)
breast$target <- as.factor(breast$target)

sel <- function(label, fs, data, target = NA) {
  out <- tryCatch({
    fitted <- fit(fs, data)
    kept <- names(transform(fitted, data))
    if (!is.na(target)) kept <- setdiff(kept, target)
    sort(kept)
  }, error = function(e) paste("ERR", conditionMessage(e)))
  cat(sprintf("R  %-22s -> %s\n", label, paste(out, collapse = ", ")))
}

cat("# diabetes (regression-style selectors)\n")
sel("corr(0.9)", feature_selection_corr(cutoff = 0.9), diabetes)
sel("lasso", feature_selection_lasso("target"), diabetes, "target")
sel("fss", feature_selection_fss("target"), diabetes, "target")
sel("stepwise(gauss)", feature_selection_stepwise("target", direction = "forward", family = stats::gaussian), diabetes, "target")

cat("# breast_cancer (classification selectors)\n")
sel("stepwise(fwd,binom)", feature_selection_stepwise("target", direction = "forward"), breast, "target")
sel("relief(top10)", feature_selection_relief("target", top = 10, m = 50, k = 10, seed = 1), breast, "target")
sel("infogain(top10)", feature_selection_info_gain("target", top = 10), breast, "target")
cat("DONE\n")
