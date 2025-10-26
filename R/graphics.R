#'@title Plot bar graph
#'@description Draw a simple bar chart from a two‑column data.frame: first column as categories (x), second as values.
#'@details If `colors` is provided, a constant fill is used; otherwise ggplot2's default palette applies.
#' `alpha` controls bar transparency. The first column is coerced to factor when needed.
#'@param data two‑column data.frame: category in the first column, numeric values in the second
#'@param label_x x‑axis label
#'@param label_y y‑axis label
#'@param colors optional fill color (single value)
#'@param alpha bar transparency (0–1)
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'#summarizing iris dataset
#'data <- iris |> dplyr::group_by(Species) |>
#' dplyr::summarize(Sepal.Length=mean(Sepal.Length))
#'head(data)
#'
#'# plotting data
#'grf <- plot_bar(data, colors="blue")
#'plot(grf)
#'@importFrom ggplot2 ggplot
#'@importFrom ggplot2 aes_string
#'@importFrom ggplot2 geom_bar
#'@importFrom ggplot2 theme_bw
#'@importFrom ggplot2 theme
#'@importFrom ggplot2 xlab
#'@importFrom ggplot2 ylab
#'@importFrom ggplot2 element_blank
#'@export
plot_bar <- function(data, label_x = "", label_y = "", colors = NULL, alpha=1) {
  series <- as.data.frame(data)
  # ensure first column is categorical for discrete bars
  if (!is.factor(series[,1]))
    series[,1] <- as.factor(series[,1])
  grf <- ggplot2::ggplot(series, ggplot2::aes_string(x=colnames(series)[1], y=colnames(series)[2]))
  if (!is.null(colors)) {
    # fixed fill color
    grf <- grf + ggplot2::geom_bar(stat = "identity", fill=colors, alpha=alpha)
  }
  else {
    # default theme colors
    grf <- grf + ggplot2::geom_bar(stat = "identity", alpha=alpha)
  }
  grf <- grf + ggplot2::theme_bw(base_size = 10)
  grf <- grf + ggplot2::theme(panel.grid.minor = ggplot2::element_blank())
  grf <- grf + ggplot2::theme(legend.title = ggplot2::element_blank()) + ggplot2::theme(legend.position = "bottom")
  grf <- grf + ggplot2::xlab(label_x)
  grf <- grf + ggplot2::ylab(label_y)
  return(grf)
}

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
  grf <- ggplot2::ggplot(data=data, ggplot2::aes(y = value, x = x))
  if (!is.null(colors)) {
    grf <- grf + ggplot2::geom_boxplot(fill=colors)
  }
  else {
    grf <- grf + ggplot2::geom_boxplot()
  }
  grf <- grf + ggplot2::labs(color=levels(data$x))
  if (!is.null(colors)) {
    grf <- grf + ggplot2::scale_fill_manual(levels(data$x), values = colors)
  }
  grf <- grf + ggplot2::theme_bw(base_size = 10)
  grf <- grf + ggplot2::theme(panel.grid.minor = ggplot2::element_blank()) + ggplot2::theme(legend.position = "bottom")
  grf <- grf + ggplot2::xlab(label_x)
  grf <- grf + ggplot2::ylab(label_y)
  return(grf)
}


#'@title Plot density
#'@description Kernel density plot for one or multiple numeric columns.
#'@details If `data` has multiple numeric columns, densities are overlaid and filled by column (group).
#' When a single column is provided, `colors` (if set) is used as a constant fill.
#' The `bin` argument is passed to `geom_density(binwidth=...)`.
#'@param data data.frame with one or more numeric columns
#'@param label_x x‑axis label
#'@param label_y y‑axis label
#'@param colors optional fill color (single column) or vector for groups
#'@param bin optional bin width passed to `geom_density`
#'@param alpha fill transparency (0–1)
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'grf <- plot_density(iris |> dplyr::select(Sepal.Width), colors="blue")
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
plot_density <- function(data, label_x = "", label_y = "", colors = NULL, bin = NULL, alpha=0.25) {
  value <- 0
  variable <- 0
  grouped <- ncol(data) > 1
  cnames <- colnames(data)
  series <- reshape::melt(as.data.frame(data))
  if (grouped) {
    grf <- ggplot2::ggplot(series, ggplot2::aes(x=value,fill=variable))
    if (is.null(bin))
      grf <- grf + ggplot2::geom_density(alpha = alpha)
    else
      grf <- grf + ggplot2::geom_density(binwidth = bin, alpha = alpha)
  }
  else {
    grf <- ggplot2::ggplot(series, ggplot2::aes(x=value))
    if (is.null(bin)) {
      if (!is.null(colors))
        grf <- grf + ggplot2::geom_density(fill=colors, alpha = alpha)
      else
        grf <- grf + ggplot2::geom_density(alpha = alpha)
    }
    else {
      if (!is.null(colors))
        grf <- grf + ggplot2::geom_density(binwidth = bin,fill=colors, alpha = alpha)
      else
        grf <- grf + ggplot2::geom_density(binwidth = bin, alpha = alpha)
    }
  }
  grf <- grf + ggplot2::theme_bw(base_size = 10)
  grf <- grf + ggplot2::xlab(label_x)
  grf <- grf + ggplot2::ylab(label_y)
  if (!is.null(colors))
    grf <- grf + ggplot2::scale_fill_manual(name = cnames, values = colors)
  grf <- grf + ggplot2::theme(panel.grid.major = ggplot2::element_blank()) + ggplot2::theme(panel.grid.minor = ggplot2::element_blank())
  grf <- grf + ggplot2::theme(legend.title = ggplot2::element_blank(), legend.position = "bottom")
  return(grf)
}

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

#'@title Plot grouped bar
#'@description Grouped (side‑by‑side) bar chart for multiple series per category.
#'@details Expects a data.frame where the first column is the category (x) and the remaining columns are
#' numeric series. Bars are grouped by series. Provide `colors` with length equal to the number of series to set fills.
#'@param data data.frame with category in first column and series in remaining columns
#'@param label_x x‑axis label
#'@param label_y y‑axis label
#'@param colors optional vector of fill colors, one per series
#'@param alpha bar transparency (0–1)
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'#summarizing iris dataset
#'data <- iris |> dplyr::group_by(Species) |>
#' dplyr::summarize(Sepal.Length=mean(Sepal.Length), Sepal.Width=mean(Sepal.Width))
#'head(data)
#'
#'#ploting data
#'grf <- plot_groupedbar(data, colors=c("blue", "red"))
#'plot(grf)
#'@importFrom ggplot2 ggplot
#'@importFrom ggplot2 aes_string
#'@importFrom ggplot2 geom_bar
#'@importFrom ggplot2 theme_bw
#'@importFrom ggplot2 theme
#'@importFrom ggplot2 xlab
#'@importFrom ggplot2 ylab
#'@importFrom ggplot2 element_blank
#'@importFrom reshape melt
#'@export
plot_groupedbar <- function(data, label_x = "", label_y = "", colors = NULL, alpha=1) {
  variable <- 0
  value <- 0
  x <- 0
  cnames <- colnames(data)[-1]
  series <- reshape::melt(as.data.frame(data), id.vars = c(1))
  colnames(series)[1] <- "x"
  if (!is.factor(series$x))
    series$x <- as.factor(series$x)

  grf <- ggplot2::ggplot(series, ggplot2::aes(x, value, fill=variable))
  grf <- grf + ggplot2::geom_bar(stat = "identity",position = "dodge", alpha=alpha)
  if (!is.null(colors)) {
    grf <- grf + ggplot2::scale_fill_manual(cnames, values = colors)
  }
  grf <- grf + ggplot2::theme_bw(base_size = 10)
  grf <- grf + ggplot2::theme(panel.grid.minor = ggplot2::element_blank())
  grf <- grf + ggplot2::theme(legend.title = ggplot2::element_blank()) + ggplot2::theme(legend.position = "bottom")
  grf <- grf + ggplot2::xlab(label_x)
  grf <- grf + ggplot2::ylab(label_y)
  return(grf)
}

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

#'@title Plot pie
#'@description Pie chart from a two‑column data.frame (category, value) using polar coordinates.
#'@details Slices are sized by the second (numeric) column. Text and border colors can be customized.
#'@param data two‑column data.frame with category and value
#'@param label_x x‑axis label (unused in pie, kept for symmetry)
#'@param label_y y‑axis label (unused in pie)
#'@param colors vector of slice fills
#'@param textcolor label text color
#'@param bordercolor slice border color
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'#summarizing iris dataset
#'data <- iris |> dplyr::group_by(Species) |>
#' dplyr::summarize(Sepal.Length=mean(Sepal.Length))
#'head(data)
#'
#'#ploting data
#'grf <- plot_pieplot(data, colors=c("red", "green", "blue"))
#'plot(grf)
#'@importFrom ggplot2 ggplot
#'@importFrom ggplot2 geom_bar
#'@importFrom ggplot2 aes
#'@importFrom ggplot2 coord_polar
#'@importFrom ggplot2 theme_light
#'@importFrom ggplot2 theme
#'@importFrom ggplot2 xlab
#'@importFrom ggplot2 ylab
#'@importFrom ggplot2 element_blank
#'@importFrom reshape melt
#'@importFrom dplyr filter
#'@importFrom dplyr group_by
#'@importFrom dplyr summarise
#'@importFrom dplyr arrange
#'@importFrom dplyr mutate
#'@export
plot_pieplot <- function(data, label_x = "", label_y = "", colors = NULL, textcolor="white", bordercolor="black") {
  x <- prop <- ypos <- label <- value <- desc <- n <- 0

  prepare.pieplot <- function(series) {
    colnames(series) <- c("x", "value")
    if (!is.factor(series$x)) {
      series$x <- as.factor(series$x)
    }

    series$colors <- colors

    series <- series |>
      dplyr::arrange(desc(x)) |>
      dplyr::mutate(prop = value / sum(series$value) *100) |>
      dplyr::mutate(ypos = cumsum(prop)- 0.5*prop) |>
      dplyr::mutate(label = paste(round(value / sum(value) * 100, 0), "%"))
    return(series)
  }
  series <- prepare.pieplot(data)

  # Basic piechart
  grf <- ggplot2::ggplot(series, ggplot2::aes(x="", y=prop, fill=x))
  grf <- grf + ggplot2::geom_bar(width = 1, stat = "identity", color=bordercolor)
  grf <- grf + ggplot2::theme_minimal(base_size = 10)
  grf <- grf + ggplot2::coord_polar("y", start=0)
  grf <- grf + ggplot2::geom_text( ggplot2::aes(y = ypos, label = label), size=6, color=textcolor)
  if (!is.null(colors))
    grf <- grf + ggplot2::scale_fill_manual(series$x, values = colors)
  grf <- grf + ggplot2::theme(panel.grid.minor = ggplot2::element_blank()) + ggplot2::theme(legend.position = "bottom")
  grf <- grf + ggplot2::xlab(label_x)
  grf <- grf + ggplot2::ylab(label_y)
  grf <- grf + ggplot2::theme(axis.text.x=ggplot2::element_blank(), legend.title = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank(), panel.grid = ggplot2::element_blank())
  return(grf)
}

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
  grf <- ggplot2::ggplot(series, ggplot2::aes_string(x=colnames(series)[1], y=colnames(series)[2], group="group")) +
    ggplot2::geom_point(size=2, color=colors) +
    ggplot2::geom_polygon(size = 1, alpha= 0.1, color=colors) +
    ggplot2::theme_light() +
    ggplot2::coord_polar()
  return(grf)
}

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

#'@title Plot stacked bar
#'@description Stacked bar chart for multiple series per category.
#'@details Expects a data.frame with category in the first column and series in remaining columns.
#' Bars are stacked within each category. Provide `colors` (one per series) to control fills.
#'@param data data.frame with category in first column and series in remaining columns
#'@param label_x x‑axis label
#'@param label_y y‑axis label
#'@param colors optional vector of fill colors, one per series
#'@param alpha bar transparency (0–1)
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'#summarizing iris dataset
#'data <- iris |> dplyr::group_by(Species) |>
#' dplyr::summarize(Sepal.Length=mean(Sepal.Length), Sepal.Width=mean(Sepal.Width))
#'
#'#plotting data
#'grf <- plot_stackedbar(data, colors=c("blue", "red"))
#'plot(grf)
#'@importFrom ggplot2 ggplot
#'@importFrom ggplot2 aes
#'@importFrom ggplot2 geom_bar
#'@importFrom ggplot2 scale_fill_manual
#'@importFrom ggplot2 theme_bw
#'@importFrom ggplot2 theme
#'@importFrom ggplot2 scale_x_discrete
#'@importFrom ggplot2 xlab
#'@importFrom ggplot2 ylab
#'@importFrom ggplot2 element_blank
#'@importFrom reshape melt
#'@export
plot_stackedbar <- function(data, label_x = "", label_y = "", colors = NULL, alpha=1) {
  x <- 0
  value <- 0
  variable <- 0
  cnames <- colnames(data)[-1]
  series <- reshape::melt(as.data.frame(data), id.vars = c(1))
  colnames(series)[1] <- "x"
  if (!is.factor(series$x))
    series$x <- as.factor(series$x)

  grf <- ggplot2::ggplot(series, ggplot2::aes(x=x, y=value, fill=variable)) + ggplot2::geom_bar(stat="identity", colour="white")
  if (!is.null(colors)) {
    grf <- grf + ggplot2::scale_fill_manual(cnames, values = colors)
  }
  grf <- grf + ggplot2::theme_bw(base_size = 10)
  grf <- grf + ggplot2::theme(panel.grid.minor = ggplot2::element_blank())
  grf <- grf + ggplot2::theme(legend.title = ggplot2::element_blank()) + ggplot2::theme(legend.position = "bottom")
  grf <- grf + ggplot2::scale_x_discrete(limits = unique(series$x))
  grf <- grf + ggplot2::xlab(label_x)
  grf <- grf + ggplot2::ylab(label_y)
  return(grf)
}

#'@title Plot time series chart
#'@description Simple time series plot with points and a line.
#'@details If `x` is NULL, an integer index 1:n is used. The color applies to both points and line.
#'@param x time index (numeric vector) or NULL to use 1:length(y)
#'@param y numeric series
#'@param label_x x‑axis label
#'@param label_y y‑axis label
#'@param color color for the series
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'x <- seq(0, 10, 0.25)
#'y <- sin(x)
#'
#'grf <- plot_ts(x = x, y = y, color=c("red"))
#'plot(grf)
#'@export
#'@importFrom ggplot2 ggplot
#'@importFrom ggplot2 aes
#'@importFrom ggplot2 geom_point
#'@importFrom ggplot2 geom_line
#'@importFrom ggplot2 theme_bw
#'@importFrom ggplot2 theme
#'@importFrom ggplot2 xlab
#'@importFrom ggplot2 ylab
#'@importFrom ggplot2 element_blank
#'@importFrom reshape melt
plot_ts <- function(x = NULL, y, label_x = "", label_y = "", color="black") {
  y <- as.vector(y)
  if (is.null(x))
    x <- 1:length(y)
  grf <- ggplot2::ggplot() +
    ggplot2::geom_point(ggplot2::aes(x = x, y = y), color = color) +
    ggplot2::geom_line( ggplot2::aes(x = x, y = y), color = color) +
    ggplot2::xlab(label_x) +
    ggplot2::ylab(label_y) +
    ggplot2::theme_bw(base_size = 10) +
    ggplot2::theme(panel.grid.major = ggplot2::element_blank()) +
    ggplot2::theme(panel.grid.minor = ggplot2::element_blank()) +
    ggplot2::theme(legend.title = ggplot2::element_blank()) +
    ggplot2::theme(legend.position = "bottom") +
    ggplot2::theme(legend.key = ggplot2::element_blank())
  return(grf)
}

#'@title Plot time series with predictions
#'@description Plot original series plus dashed lines for in‑sample adjustment and optional out‑of‑sample predictions.
#'@details `yadj` length defines the training segment; `ypred` (if provided) is appended after `yadj`.
#'@param x time index (numeric vector) or NULL to use 1:length(y)
#'@param y numeric time series
#'@param yadj fitted/adjusted values for the training window
#'@param ypred optional predicted values after the training window
#'@param label_x x‑axis title
#'@param label_y y‑axis title
#'@param color color for the original series
#'@param color_adjust color for the adjusted values (dashed)
#'@param color_prediction color for the predictions (dashed)
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'x <- base::seq(0, 10, 0.25)
#'yvalues <- sin(x) + rnorm(41,0,0.1)
#'adjust <- sin(x[1:35])
#'prediction <- sin(x[36:41])
#'grf <- plot_ts_pred(y=yvalues, yadj=adjust, ypred=prediction)
#'plot(grf)
#'@export
#'@importFrom ggplot2 ggplot
#'@importFrom ggplot2 aes
#'@importFrom ggplot2 geom_point
#'@importFrom ggplot2 geom_line
#'@importFrom ggplot2 scale_fill_manual
#'@importFrom ggplot2 theme_bw
#'@importFrom ggplot2 theme
#'@importFrom ggplot2 xlab
#'@importFrom ggplot2 ylab
#'@importFrom ggplot2 element_blank
#'@importFrom reshape melt
plot_ts_pred <- function(x = NULL, y, yadj, ypred = NULL, label_x = "", label_y = "", color="black", color_adjust="blue", color_prediction="green") {
  y <- as.vector(y)
  if (is.null(x))
    x <- 1:length(y)
  y <- as.vector(y)
  yadj <- as.vector(yadj)
  ntrain <- length(yadj)
  yhat <- yadj
  ntest <- 0
  if (!is.null(ypred)) {
    ypred <- as.vector(ypred)
    yhat <- c(yhat, ypred)
    ntest <- length(ypred)
  }

  grf <- ggplot2::ggplot() +
    ggplot2::geom_point( ggplot2::aes(x = x, y = y), color = color) +
    ggplot2::geom_line( ggplot2::aes(x = x, y = y), color = color) +
    ggplot2::xlab(label_x) +
    ggplot2::ylab(label_y) +
    ggplot2::theme_bw(base_size = 10) +
    ggplot2::theme(panel.grid.major = ggplot2::element_blank()) +
    ggplot2::theme(panel.grid.minor = ggplot2::element_blank()) +
    ggplot2::theme(legend.title = ggplot2::element_blank()) +
    ggplot2::theme(legend.position = "bottom") +
    ggplot2::theme(legend.key = ggplot2::element_blank())

  grf <- grf + ggplot2::geom_line( ggplot2::aes(x = x[1:ntrain], y = yhat[1:ntrain]),
                                   color = color_adjust, linetype = "dashed")
  if (!is.null(ypred))
    grf <- grf +ggplot2::geom_line( ggplot2::aes(x = x[ntrain:(ntrain+ntest)], y = yhat[ntrain:(ntrain+ntest)]),
                                    color = color_prediction, linetype = "dashed")
  return(grf)
}
#' Graphics utilities
#' @description A collection of small plotting helpers built on ggplot2 used across the package
#' to quickly visualize vectors, grouped summaries and time series. All functions return a
#' `ggplot2::ggplot` object so you can further customize the theme, scales, and annotations.
#' @details
#' Conventions adopted:
#' - Input data generally follows the pattern: first column is an index or category (x), remaining columns
#'   are numeric series; in some functions a long format is expected with columns named `x`, `value`, `variable`.
#' - The `colors` parameter accepts either a single color or a vector mapped to groups/variables.
#' - Transparency is controlled by `alpha` where provided.
#' - All helpers set a light `theme_bw()` baseline and place legends at the bottom by default.
#' @keywords visualization graphics
#' @name dal_graphics
#' @seealso ggplot2
NULL
