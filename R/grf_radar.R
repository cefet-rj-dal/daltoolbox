#'@title Plot radar
#'@description Radar (spider) chart for a single profile of variables using polar coordinates.
#'@details Expects a two‑column data.frame with variable names in the first column and numeric values in the second.
#'@param data two‑column data.frame: variable name and value
#'@param label_x x‑axis label (unused; variable names are shown around the circle)
#'@param label_y y‑axis label
#'@param colors line/fill color for the polygon
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'data <- data.frame(name = "Petal.Length", value = mean(iris$Petal.Length))
#'data <- rbind(data, data.frame(name = "Petal.Width", value = mean(iris$Petal.Width)))
#'data <- rbind(data, data.frame(name = "Sepal.Length", value = mean(iris$Sepal.Length)))
#'data <- rbind(data, data.frame(name = "Sepal.Width", value = mean(iris$Sepal.Width)))
#'
#'grf <- plot_radar(data, colors="red") + ggplot2::ylim(0, NA)
#'plot(grf)
#'@importFrom ggplot2 ggplot
#'@importFrom ggplot2 aes
#'@importFrom ggplot2 geom_boxplot
#'@importFrom ggplot2 scale_fill_manual
#'@importFrom ggplot2 theme_bw
#'@importFrom ggplot2 geom_point
#'@importFrom ggplot2 geom_polygon
#'@importFrom ggplot2 theme
#'@importFrom ggplot2 xlab
#'@importFrom ggplot2 ylab
#'@importFrom reshape melt
#'@export
plot_radar <- function(data, label_x = "", label_y = "", colors = NULL) {
  series <- as.data.frame(data)
  if (!is.factor(series[,1]))
    series[,1] <- as.factor(series[,1])
  series$group <- 1
  x <- y <- group <- NULL
  colnames(series)[1:2] <- c("x", "y")
  grf <- ggplot2::ggplot(series, ggplot2::aes(x = x, y = y, group = group)) +
    ggplot2::geom_point(size=2, color=colors) +
    ggplot2::geom_polygon(linewidth = 1, alpha= 0.1, color=colors) +
    ggplot2::theme_light() +
    ggplot2::coord_polar()
  return(grf)
}

