#'@title Plot points
#'@description Dot chart for multiple series across categories (points only).
#'@details Expects a data.frame with category in the first column and one or more numeric series.
#' Points are colored by series (legend shows original column names). Supply `colors` to override the palette.
#'@param data data.frame with category + one or more numeric columns
#'@param label_x x‑axis label
#'@param label_y y‑axis label
#'@param colors optional color vector for series
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'x <- seq(0, 10, 0.25)
#'data <- data.frame(x, sin=sin(x), cosine=cos(x)+5)
#'head(data)
#'
#'grf <- plot_points(data, colors=c("red", "green"))
#'plot(grf)
#'@importFrom ggplot2 ggplot
#'@importFrom ggplot2 geom_point
#'@importFrom ggplot2 aes
#'@importFrom ggplot2 scale_color_manual
#'@importFrom ggplot2 theme_light
#'@importFrom ggplot2 theme
#'@importFrom ggplot2 xlab
#'@importFrom ggplot2 ylab
#'@importFrom ggplot2 element_blank
#'@importFrom reshape melt
#'@export
plot_points <- function(data, label_x = "", label_y = "", colors = NULL) {
  x <- 0
  value <- 0
  variable <- 0
  series <- reshape::melt(as.data.frame(data), id.vars = c(1))
  cnames <- colnames(data)[-1]
  colnames(series)[1] <- "x"
  grf <- ggplot2::ggplot(data=series, ggplot2::aes(x = x, y = value, colour=variable, group=variable)) +
    ggplot2::geom_point(size=1)
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

