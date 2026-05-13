#' @title Graphics Utilities
#' @description A collection of small plotting helpers built on ggplot2 used
#'   across the package to quickly visualize vectors, grouped summaries, and
#'   time series.
#' @details All functions return a `ggplot2::ggplot` object so the result can be
#'   extended with themes, scales, labels, and annotations.
#'
#' Conventions adopted:
#' - Input data generally follows the pattern: first column is an index or
#'   category (`x`), remaining columns are numeric series.
#' - Some functions expect a long format with columns named `x`, `value`, and
#'   `variable`.
#' - The `colors` parameter accepts either a single color or a vector mapped to
#'   groups or variables.
#' - Transparency is controlled by `alpha` where provided.
#' - All helpers set a light `theme_bw()` baseline and place legends at the
#'   bottom by default.
#' @examples
#' data <- data.frame(group = c("A", "B"), value = c(2, 5))
#' grf <- plot_bar(data)
#' plot(grf)
#' @keywords visualization graphics
#' @name dal_graphics
#' @seealso ggplot2
NULL
