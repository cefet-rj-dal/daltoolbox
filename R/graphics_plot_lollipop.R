#'@title Plot lollipop
#'@description Lollipop chart (stick + circle + value label) per category.
#'@details Expects a data.frame with category in the first column and numeric values in subsequent columns.
#' Circles are drawn at values, with vertical segments extending from `min_value` to `value - max_value_gap`.
#'@param data data.frame with category and numeric values
#'@param label_x x‑axis label
#'@param label_y y‑axis label
#'@param colors stick/circle color
#'@param color_text color of the text inside the circle
#'@param size_text text size
#'@param size_ball circle size
#'@param alpha_ball circle transparency (0–1)
#'@param min_value minimum baseline for the stick
#'@param max_value_gap gap from value to stick end
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'#summarizing iris dataset
#'data <- iris |> dplyr::group_by(Species) |>
#' dplyr::summarize(Sepal.Length=mean(Sepal.Length))
#'head(data)
#'
#'#ploting data
#'grf <- plot_lollipop(data, colors="blue", max_value_gap=0.2)
#'plot(grf)
#'@importFrom ggplot2 ggplot
#'@importFrom ggplot2 aes
#'@importFrom ggplot2 geom_segment
#'@importFrom ggplot2 geom_text
#'@importFrom ggplot2 geom_point
#'@importFrom ggplot2 theme_light
#'@importFrom ggplot2 geom_histogram
#'@importFrom ggplot2 theme_bw
#'@importFrom ggplot2 theme
#'@importFrom ggplot2 xlab
#'@importFrom ggplot2 ylab
#'@importFrom ggplot2 element_blank
#'@importFrom reshape melt
#'@export
plot_lollipop <- function(data, label_x = "", label_y = "", colors = NULL, color_text = "black", size_text=3, size_ball=8, alpha_ball=0.2, min_value=0, max_value_gap=1) {
  value <- 0
  x <- 0
  cnames <- colnames(data)[-1]
  data <- reshape::melt(as.data.frame(data), id.vars = c(1))
  colnames(data)[1] <- "x"
  if (!is.factor(data$x))
    data$x <- as.factor(data$x)
  data$value <- round(data$value)

  grf <- ggplot2::ggplot(data=data, ggplot2::aes(x=x, y=value, label=value)) +
    ggplot2::geom_segment( ggplot2::aes(x=x, xend=x, y=min_value, yend=(value-max_value_gap)), color=colors, size=1) +
    ggplot2::geom_text(color=color_text, size=size_text) +
    ggplot2::geom_point(color=colors, size=size_ball, alpha=alpha_ball) +
    ggplot2::theme_light() +
    ggplot2::theme(
      panel.grid.major.y = ggplot2::element_blank(),
      panel.border = ggplot2::element_blank(),
      axis.ticks.y = ggplot2::element_blank()
    ) +
    ggplot2::ylab(label_y) + ggplot2::xlab(label_x)
  return(grf)
}

