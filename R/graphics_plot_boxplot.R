#'@title Plot boxplot
#'@description Boxplots for each numeric column of a data.frame.
#'@details The data is melted to long format and a box is drawn per original column. If `colors` is provided,
#' a constant fill is applied to all boxes. Use `barwidth` to control box width.
#'@param data data.frame with one or more numeric columns
#'@param label_x x‑axis label
#'@param label_y y‑axis label
#'@param colors optional fill color for boxes
#'@param barwidth width of the box (numeric)
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'grf <- plot_boxplot(iris, colors="white")
#'plot(grf)
#'@importFrom ggplot2 ggplot
#'@importFrom ggplot2 aes
#'@importFrom ggplot2 geom_boxplot
#'@importFrom ggplot2 scale_fill_manual
#'@importFrom ggplot2 labs
#'@importFrom ggplot2 theme_bw
#'@importFrom ggplot2 theme
#'@importFrom ggplot2 xlab
#'@importFrom ggplot2 ylab
#'@importFrom ggplot2 element_blank
#'@importFrom reshape melt
#'@export
plot_boxplot <- function(data, label_x = "", label_y = "", colors = NULL, barwidth=0.25) {
  value <- 0
  variable <- 0
  cnames <- colnames(data)
  series <- reshape::melt(as.data.frame(data))
  grf <- ggplot2::ggplot( ggplot2::aes(y = value, x = variable), data = series)
  if (!is.null(colors)) {
    grf <- grf + ggplot2::geom_boxplot(fill = colors, width=barwidth)
  }
  else {
    grf <- grf + ggplot2::geom_boxplot(width=barwidth)
  }
  grf <- grf + ggplot2::labs(color=cnames)
  if (!is.null(colors)) {
    grf <- grf + ggplot2::scale_fill_manual(cnames, values = colors)
  }
  grf <- grf + ggplot2::theme_bw(base_size = 10)
  grf <- grf + ggplot2::theme(panel.grid.minor = ggplot2::element_blank()) + ggplot2::theme(legend.position = "bottom")
  grf <- grf + ggplot2::xlab(label_x)
  grf <- grf + ggplot2::ylab(label_y)
  return(grf)
}

