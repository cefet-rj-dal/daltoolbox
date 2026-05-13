#'@title Plot time series chart
#'@description Simple time series plot with points and a line.
#'@details If `x` is NULL, an integer index 1:n is used. The color applies to both points and line.
#'@param x time index (numeric vector) or NULL to use 1:length(y)
#'@param y numeric series
#'@param label_x x‑axis label
#'@param label_y y‑axis label
#'@param color color for the series
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'x <- seq(0, 10, 0.25)
#'y <- sin(x)
#'
#'grf <- plot_ts(x = x, y = y, color=c("red"))
#'plot(grf)
#'@export
#'@importFrom ggplot2 ggplot
#'@importFrom ggplot2 aes
#'@importFrom ggplot2 geom_point
#'@importFrom ggplot2 geom_line
#'@importFrom ggplot2 theme_bw
#'@importFrom ggplot2 theme
#'@importFrom ggplot2 xlab
#'@importFrom ggplot2 ylab
#'@importFrom ggplot2 element_blank
#'@importFrom reshape melt
plot_ts <- function(x = NULL, y, label_x = "", label_y = "", color="black") {
  y <- as.vector(y)
  if (is.null(x))
    x <- 1:length(y)
  grf <- ggplot2::ggplot() +
    ggplot2::geom_point(ggplot2::aes(x = x, y = y), color = color) +
    ggplot2::geom_line( ggplot2::aes(x = x, y = y), color = color) +
    ggplot2::xlab(label_x) +
    ggplot2::ylab(label_y) +
    ggplot2::theme_bw(base_size = 10) +
    ggplot2::theme(panel.grid.major = ggplot2::element_blank()) +
    ggplot2::theme(panel.grid.minor = ggplot2::element_blank()) +
    ggplot2::theme(legend.title = ggplot2::element_blank()) +
    ggplot2::theme(legend.position = "bottom") +
    ggplot2::theme(legend.key = ggplot2::element_blank())
  return(grf)
}

