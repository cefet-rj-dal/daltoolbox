#'@title Plot series
#'@description Line plot for one or more series over a common x index.
#'@details Expects a data.frame where the first column is the x index and remaining columns are numeric series.
#' Points and lines are drawn per series; supply `colors` to override the palette.
#'@param data data.frame with x in the first column and series in remaining columns
#'@param label_x x‑axis label
#'@param label_y y‑axis label
#'@param colors optional vector of colors for series
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'x <- seq(0, 10, 0.25)
#'data <- data.frame(x, sin=sin(x))
#'head(data)
#'
#'grf <- plot_series(data, colors=c("red"))
#'plot(grf)
#'@importFrom ggplot2 ggplot
#'@importFrom ggplot2 aes
#'@importFrom ggplot2 geom_point
#'@importFrom ggplot2 geom_line
#'@importFrom ggplot2 scale_color_manual
#'@importFrom ggplot2 scale_fill_manual
#'@importFrom ggplot2 theme_bw
#'@importFrom ggplot2 theme
#'@importFrom ggplot2 labs
#'@importFrom ggplot2 xlab
#'@importFrom ggplot2 ylab
#'@importFrom ggplot2 element_blank
#'@importFrom reshape melt
#'@export
plot_series <- function(data, label_x = "", label_y = "", colors = NULL) {
  x <- 0
  value <- 0
  variable <- 0
  series <- reshape::melt(as.data.frame(data), id.vars = c(1))
  cnames <- colnames(data)[-1]
  colnames(series)[1] <- "x"
  grf <- ggplot2::ggplot(data=series, ggplot2::aes(x = x, y = value, colour=variable, group=variable)) +
    ggplot2::geom_point(size=1.5) +
    ggplot2::geom_line(linewidth=1)
  if (!is.null(colors)) {
    grf <- grf + ggplot2::scale_color_manual(values=colors)
  }
  grf <- grf + ggplot2::labs(color=cnames)
  grf <- grf + ggplot2::xlab(label_x)
  grf <- grf + ggplot2::ylab(label_y)
  grf <- grf + ggplot2::theme_bw(base_size = 10)
  grf <- grf + ggplot2::theme(panel.grid.major = ggplot2::element_blank()) + ggplot2::theme(panel.grid.minor = ggplot2::element_blank())
  grf <- grf + ggplot2::theme(legend.title = ggplot2::element_blank()) + ggplot2::theme(legend.position = "bottom") + ggplot2::theme(legend.key = ggplot2::element_blank())
  return(grf)
}

