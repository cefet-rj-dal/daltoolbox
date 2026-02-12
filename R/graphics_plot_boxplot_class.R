#'@title Boxplot per class
#'@description Boxplots of a numeric column grouped by a class label.
#'@details Expects a data.frame with the grouping column named in `class_label` and one numeric column.
#' The function melts to long format and draws per‑group distributions.
#'@param data data.frame with a grouping column and one numeric column
#'@param class_label name of the grouping (class) column
#'@param label_x x‑axis label
#'@param label_y y‑axis label
#'@param colors optional fill color for the boxes
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'grf <- plot_boxplot_class(iris |> dplyr::select(Sepal.Width, Species),
#' class_label = "Species", colors=c("red", "green", "blue"))
#'plot(grf)
#'@importFrom ggplot2 ggplot
#'@importFrom ggplot2 aes
#'@importFrom ggplot2 geom_boxplot
#'@importFrom ggplot2 scale_fill_manual
#'@importFrom ggplot2 theme_bw
#'@importFrom ggplot2 theme
#'@importFrom ggplot2 xlab
#'@importFrom ggplot2 ylab
#'@importFrom ggplot2 element_blank
#'@importFrom reshape melt
#'@export
plot_boxplot_class <- function(data, class_label, label_x = "", label_y = "", colors = NULL) {
  value <- 0
  variable <- 0
  x <- 0
  data <- reshape::melt(data, id=class_label)
  colnames(data)[1] <- "x"
  if (!is.factor(data$x))
    data$x <- as.factor(data$x)
  if (!is.null(colors) && length(colors) > 1) {
    grf <- ggplot2::ggplot(data = data, ggplot2::aes(y = value, x = x, fill = x))
    grf <- grf + ggplot2::geom_boxplot()
    grf <- grf + ggplot2::scale_fill_manual(values = colors)
  } else {
    grf <- ggplot2::ggplot(data = data, ggplot2::aes(y = value, x = x))
    if (!is.null(colors)) {
      grf <- grf + ggplot2::geom_boxplot(fill = colors[1])
    } else {
      grf <- grf + ggplot2::geom_boxplot()
    }
  }
  grf <- grf + ggplot2::theme_bw(base_size = 10)
  grf <- grf + ggplot2::theme(panel.grid.minor = ggplot2::element_blank()) + ggplot2::theme(legend.position = "bottom")
  grf <- grf + ggplot2::xlab(label_x)
  grf <- grf + ggplot2::ylab(label_y)
  return(grf)
}


