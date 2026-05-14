#'@title Plot radar
#'@description Radar (spider) chart for a single profile of variables using radial axes.
#'@details Expects a two‑column data.frame with variable names in the first column and numeric values in the second.
#' The graphic is built as an n-sided polygon, where n is the number of variables, so at least three
#' variables are required. The function already sets the drawing limits for the full polygon; adding
#' `ylim()` or other Cartesian clipping after the fact can hide part of the radar.
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
#'grf <- plot_radar(data, colors = "red")
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
  if (ncol(series) < 2) {
    stop("'data' must have at least two columns: variable name and value")
  }

  colnames(series)[1:2] <- c("label", "value")
  series <- series[, 1:2, drop = FALSE]
  series$label <- as.character(series$label)
  series$value <- as.numeric(series$value)
  if (any(is.na(series$value))) {
    stop("'data' second column must be numeric")
  }
  if (nrow(series) < 3) {
    stop("'data' must contain at least three variables for a radar chart")
  }

  if (is.null(colors)) {
    colors <- "black"
  }

  n_axes <- nrow(series)
  angles <- seq(pi / 2, pi / 2 - 2 * pi, length.out = n_axes + 1)[1:n_axes]
  radius_max <- max(series$value)
  grid_breaks <- pretty(c(0, radius_max), n = 5)
  grid_breaks <- grid_breaks[grid_breaks >= 0]
  if (length(grid_breaks) == 0 || max(grid_breaks) == 0) {
    grid_breaks <- c(0, 1)
  }
  radius_max <- max(grid_breaks)
  label_radius <- radius_max * 1.08
  plot_limit <- label_radius * 1.12

  series$x <- series$value * cos(angles)
  series$y <- series$value * sin(angles)
  polygon <- rbind(series, series[1, , drop = FALSE])

  axes <- data.frame(
    x = 0,
    y = 0,
    xend = radius_max * cos(angles),
    yend = radius_max * sin(angles)
  )

  label_data <- data.frame(
    x = label_radius * cos(angles),
    y = label_radius * sin(angles),
    label = series$label
  )

  grid <- do.call(rbind, lapply(grid_breaks[grid_breaks > 0], function(r) {
    theta <- seq(0, 2 * pi, length.out = 361)
    data.frame(
      x = r * cos(theta),
      y = r * sin(theta),
      group = sprintf("r_%s", format(r, trim = TRUE))
    )
  }))

  grid_labels <- data.frame(
    x = 0,
    y = grid_breaks[grid_breaks > 0],
    label = grid_breaks[grid_breaks > 0]
  )

  x <- y <- xend <- yend <- group <- label <- NULL
  grf <- ggplot2::ggplot() +
    ggplot2::geom_path(
      data = grid,
      ggplot2::aes(x = x, y = y, group = group),
      color = "grey85"
    ) +
    ggplot2::geom_segment(
      data = axes,
      ggplot2::aes(x = x, y = y, xend = xend, yend = yend),
      color = "grey85"
    ) +
    ggplot2::geom_polygon(
      data = polygon,
      ggplot2::aes(x = x, y = y),
      linewidth = 1,
      alpha = 0.1,
      fill = colors,
      color = colors
    ) +
    ggplot2::geom_path(
      data = polygon,
      ggplot2::aes(x = x, y = y),
      linewidth = 1,
      color = colors
    ) +
    ggplot2::geom_point(
      data = series,
      ggplot2::aes(x = x, y = y),
      size = 2,
      color = colors
    ) +
    ggplot2::geom_text(
      data = label_data,
      ggplot2::aes(x = x, y = y, label = label),
      color = "grey30"
    ) +
    ggplot2::geom_text(
      data = grid_labels,
      ggplot2::aes(x = x, y = y, label = label),
      color = "grey50",
      hjust = -0.2,
      size = 3
    ) +
    ggplot2::coord_equal(
      xlim = c(-plot_limit, plot_limit),
      ylim = c(-plot_limit, plot_limit),
      clip = "off"
    ) +
    ggplot2::theme_void() +
    ggplot2::theme(
      plot.background = ggplot2::element_rect(fill = "white", color = NA),
      panel.background = ggplot2::element_rect(fill = "white", color = NA),
      plot.margin = ggplot2::margin(20, 40, 20, 40)
    ) +
    ggplot2::xlab(label_x) +
    ggplot2::ylab(label_y)
  return(grf)
}

