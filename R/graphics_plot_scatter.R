#'@title Scatter graph
#'@description Scatter plot from a long data.frame with columns named `x`, `value`, and `variable`.
#'@details Colors are mapped to `variable`. If `variable` is numeric, a gradient color scale is used when `colors` is provided.
#'@param data long data.frame with columns `x`, `value`, `variable`
#'@param label_x x‑axis label
#'@param label_y y‑axis label
#'@param colors optional color(s); for numeric `variable`, supply a gradient as c(low, high)
#'@return return a ggplot2::ggplot graphic
#'@examples
#'grf <- plot_scatter(iris |> dplyr::select(x = Sepal.Length,
#' value = Sepal.Width, variable = Species),
#' label_x = "Sepal.Length", label_y = "Sepal.Width",
#' colors=c("red", "green", "blue"))
#' plot(grf)
#'@importFrom ggplot2 ggplot
#'@importFrom ggplot2 aes
#'@importFrom ggplot2 geom_point
#'@importFrom ggplot2 scale_color_manual
#'@importFrom ggplot2 scale_color_gradient
#'@importFrom ggplot2 scale_fill_manual
#'@importFrom ggplot2 theme_bw
#'@importFrom ggplot2 theme
#'@importFrom ggplot2 xlab
#'@importFrom ggplot2 ylab
#'@importFrom ggplot2 element_blank
#'@importFrom reshape melt
#'@export
plot_scatter <- function(data, label_x = "", label_y = "", colors = NULL) {
  x <- 0
  value <- 0
  variable <- 0
  grf <- ggplot2::ggplot(data=data, ggplot2::aes(x = x, y = value, colour=variable, group=variable)) +
    ggplot2::geom_point(size=1)
  if (!is.null(colors)) {
    grf <- grf + ggplot2::scale_color_manual(values=colors)
    if (!is.null(data$variable) && !is.factor(data$variable))
      grf <- grf + ggplot2::scale_color_gradient(low=colors[1], high=colors[length(colors)])
  }
  grf <- grf + ggplot2::xlab(label_x)
  grf <- grf + ggplot2::ylab(label_y)
  grf <- grf + ggplot2::theme_bw(base_size = 10)
  grf <- grf + ggplot2::theme(panel.grid.major = ggplot2::element_blank()) + ggplot2::theme(panel.grid.minor = ggplot2::element_blank())
  grf <- grf + ggplot2::theme(legend.position = "bottom") + ggplot2::theme(legend.key = ggplot2::element_blank())
  return(grf)
}

