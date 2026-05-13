#'@title Plot histogram
#'@description Histogram for a numeric column using ggplot2.
#'@details If multiple columns are provided, only the first is used. Breaks are computed via `graphics::hist` to
#' mirror base R binning. `color` controls the fill; `alpha` the transparency.
#'@param data data.frame with one numeric column (first column is used if multiple)
#'@param label_x x‑axis label
#'@param label_y y‑axis label
#'@param color fill color
#'@param alpha transparency level (0–1)
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'grf <- plot_hist(iris |> dplyr::select(Sepal.Width), color=c("blue"))
#'plot(grf)
#'@importFrom ggplot2 ggplot
#'@importFrom ggplot2 aes_string
#'@importFrom ggplot2 geom_histogram
#'@importFrom ggplot2 theme_bw
#'@importFrom ggplot2 theme
#'@importFrom ggplot2 xlab
#'@importFrom ggplot2 ylab
#'@importFrom reshape melt
#'@importFrom ggplot2 element_blank
#'@importFrom graphics hist
#'@importFrom dplyr filter summarise group_by arrange mutate
#'@export
plot_hist <- function(data, label_x = "", label_y = "", color = 'white', alpha=0.25) {
  variable <- 0
  value <- 0
  cnames <- colnames(data)[1]
  series <- reshape::melt(as.data.frame(data))
  series <- series |> dplyr::filter(variable %in% cnames)
  tmp <- graphics::hist(series$value, plot = FALSE)
  grf <- ggplot2::ggplot(series, ggplot2::aes(x=value))
  grf <- grf + ggplot2::geom_histogram(breaks=tmp$breaks,fill=color, alpha = alpha, colour="black")
  grf <- grf + ggplot2::xlab(label_x)
  grf <- grf + ggplot2::ylab(label_y)
  grf <- grf + ggplot2::theme_bw(base_size = 10)
  grf <- grf + ggplot2::scale_fill_manual(name = cnames, values = color)
  grf <- grf + ggplot2::theme(panel.grid.major = ggplot2::element_blank()) + ggplot2::theme(panel.grid.minor = ggplot2::element_blank()) + ggplot2::theme(legend.position = "bottom")
  return(grf)
}

