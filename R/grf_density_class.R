#'@title Plot density per class
#'@description Kernel density plot grouped by a class label.
#'@details Expects `data` with a grouping column named in `class_label` and one numeric column. Each group is
#' filled with a distinct color (if provided).
#'@param data data.frame with class label and a numeric column
#'@param class_label name of the grouping (class) column
#'@param label_x x‑axis label
#'@param label_y y‑axis label
#'@param colors optional vector of fills per class
#'@param bin optional bin width passed to `geom_density`
#'@param alpha fill transparency (0–1)
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'grf <- plot_density_class(iris |> dplyr::select(Sepal.Width, Species),
#' class = "Species", colors=c("red", "green", "blue"))
#'plot(grf)
#'@importFrom ggplot2 ggplot
#'@importFrom ggplot2 aes
#'@importFrom ggplot2 geom_density
#'@importFrom ggplot2 scale_fill_manual
#'@importFrom ggplot2 theme_bw
#'@importFrom ggplot2 theme
#'@importFrom ggplot2 xlab
#'@importFrom ggplot2 ylab
#'@importFrom ggplot2 element_blank
#'@importFrom reshape melt
#'@export
plot_density_class <- function(data, class_label, label_x = "", label_y = "", colors = NULL, bin = NULL, alpha=0.5) {
  value <- 0
  variable <- 0
  x <- 0
  data <- reshape::melt(data, id=class_label)
  colnames(data)[1] <- "x"
  if (!is.factor(data$x))
    data$x <- as.factor(data$x)
  grf <- ggplot2::ggplot(data=data, ggplot2::aes(x = value, fill = x))
  if (is.null(bin))
    grf <- grf + ggplot2::geom_density(alpha = alpha)
  else
    grf <- grf + ggplot2::geom_density(binwidth = bin, alpha = alpha)
  grf <- grf + ggplot2::theme_bw(base_size = 10)
  grf <- grf + ggplot2::xlab(label_x)
  grf <- grf + ggplot2::ylab(label_y)
  if (!is.null(colors))
    grf <- grf + ggplot2::scale_fill_manual(name = levels(data$x), values = colors)
  grf <- grf + ggplot2::theme(panel.grid.major = ggplot2::element_blank()) + ggplot2::theme(panel.grid.minor = ggplot2::element_blank())
  grf <- grf + ggplot2::theme(legend.title = ggplot2::element_blank(), legend.position = "bottom")
  return(grf)
}

