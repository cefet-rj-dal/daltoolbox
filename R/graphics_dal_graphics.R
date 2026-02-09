#' Graphics utilities
#' @description A collection of small plotting helpers built on ggplot2 used across the package
#' to quickly visualize vectors, grouped summaries and time series. All functions return a
#' `ggplot2::ggplot` object so you can further customize the theme, scales, and annotations.
#' @details
#' Conventions adopted:
#' - Input data generally follows the pattern: first column is an index or category (x), remaining columns
#'   are numeric series; in some functions a long format is expected with columns named `x`, `value`, `variable`.
#' - The `colors` parameter accepts either a single color or a vector mapped to groups/variables.
#' - Transparency is controlled by `alpha` where provided.
#' - All helpers set a light `theme_bw()` baseline and place legends at the bottom by default.
#' @keywords visualization graphics
#' @name dal_graphics
#' @seealso ggplot2
NULL
