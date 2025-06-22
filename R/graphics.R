#'@title Plot bar graph
#'@description this function displays a bar graph from a data frame containing x-axis categories using ggplot2.
#'@param data data.frame contain x, value, and variable
#'@param label_x x-axis label
#'@param label_y y-axis label
#'@param colors color vector
#'@param alpha level of transparency
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'#summarizing iris dataset
#'data <- iris |> dplyr::group_by(Species) |>
#' dplyr::summarize(Sepal.Length=mean(Sepal.Length))
#'head(data)
#'
#'#ploting data
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
  if (!is.factor(series[,1]))
    series[,1] <- as.factor(series[,1])
  grf <- ggplot2::ggplot(series, ggplot2::aes_string(x=colnames(series)[1], y=colnames(series)[2]))
  if (!is.null(colors)) {
    grf <- grf + ggplot2::geom_bar(stat = "identity", fill=colors, alpha=alpha)
  }
  else {
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
#'@description this function displays a boxplot graph from a data frame containing x-axis categories and numeric values using ggplot2.
#'@param data data.frame contain x, value, and variable
#'@param label_x x-axis label
#'@param label_y y-axis label
#'@param colors color vector
#'@param barwidth width of bar
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
#'@description This function generates boxplots grouped by a specified class label from a data frame containing numeric values using ggplot2.
#'@param data data.frame contain x, value, and variable
#'@param class_label name of attribute for class label
#'@param label_x x-axis label
#'@param label_y y-axis label
#'@param colors color vector
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'grf <- plot_boxplot_class(iris |> dplyr::select(Sepal.Width, Species),
#' class = "Species", colors=c("red", "green", "blue"))
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
#'@description This function generates a density plot from a data frame containing numeric values using ggplot2.
#'If the data frame has multiple columns, densities can be grouped and plotted.
#'@param data data.frame contain x, value, and variable
#'@param label_x x-axis label
#'@param label_y y-axis label
#'@param colors color vector
#'@param bin bin width for density estimation
#'@param alpha level of transparency
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
#'@description This function generates density plots using ggplot2 grouped by a specified class label from a data frame containing numeric values.
#'@param data data.frame contain x, value, and variable
#'@param class_label name of attribute for class label
#'@param label_x x-axis label
#'@param label_y y-axis label
#'@param colors color vector
#'@param bin bin width for density estimation
#'@param alpha level of transparency
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
#'@description This function generates a grouped bar plot from a given data frame using ggplot2.
#'@param data data.frame contain x, value, and variable
#'@param label_x x-axis label
#'@param label_y y-axis label
#'@param colors color vector
#'@param alpha level of transparency
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
#'@description This function generates a histogram from a specified data frame using ggplot2.
#'@param data data.frame contain x, value, and variable
#'@param label_x x-axis label
#'@param label_y y-axis label
#'@param color color vector
#'@param alpha transparency level
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
#'@description This function creates a lollipop chart using ggplot2.
#'@param data data.frame contain x, value, and variable
#'@param label_x x-axis label
#'@param label_y y-axis label
#'@param colors color vector
#'@param color_text color of text inside ball
#'@param size_text size of text inside ball
#'@param size_ball size of ball
#'@param alpha_ball transparency of ball
#'@param min_value minimum value
#'@param max_value_gap maximum value gap
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
#'@description This function creates a pie chart using ggplot2.
#'@param data data.frame contain x, value, and variable
#'@param label_x x-axis label
#'@param label_y y-axis label
#'@param colors color vector
#'@param textcolor text color
#'@param bordercolor border color
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
#'@description This function creates a scatter plot using ggplot2.
#'@param data data.frame contain x, value, and variable
#'@param label_x x-axis label
#'@param label_y y-axis label
#'@param colors color vector
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
#'@description This function creates a radar chart using ggplot2.
#'@param data data.frame contain x, value, and variable
#'@param label_x x-axis label
#'@param label_y y-axis label
#'@param colors color vector
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
#'@description This function creates a scatter plot using ggplot2.
#'@param data data.frame contain x, value, and variable
#'@param label_x x-axis label
#'@param label_y y-axis label
#'@param colors color vector
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
#'@description This function creates a time series plot using ggplot2.
#'@param data data.frame contain x, value, and variable
#'@param label_x x-axis label
#'@param label_y y-axis label
#'@param colors color vector
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
#'@description this function creates a stacked bar chart using ggplot2.
#'@param data data.frame contain x, value, and variable
#'@param label_x x-axis label
#'@param label_y y-axis label
#'@param colors color vector
#'@param alpha level of transparency
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
#'@description This function plots a time series chart with points and a line using ggplot2.
#'@param x input variable
#'@param y output variable
#'@param label_x x-axis label
#'@param label_y y-axis label
#'@param color color for time series
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'x <- seq(0, 10, 0.25)
#'data <- data.frame(x, sin=sin(x))
#'head(data)
#'
#'grf <- plot_ts(x = data$x, y = data$sin, color=c("red"))
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

#'@title Plot a time series chart with predictions
#'@description This function plots a time series chart with three lines: the original series, the adjusted series, and the predicted series using ggplot2.
#'@param x time index
#'@param y time series
#'@param yadj adjustment of time series
#'@param ypred prediction of the time series
#'@param label_x x-axis title
#'@param label_y y-axis title
#'@param color color for the time series
#'@param color_adjust color for the adjusted values
#'@param color_prediction color for the predictions
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'x <- base::seq(0, 10, 0.25)
#'yvalues <- sin(x) + rnorm(41,0,0.1)
#'adjust <- sin(x[1:35])
#'prediction <- sin(x[36:41])
#'grf <- plot_ts_pred(y=yvalues, yadj=adjust, ypre=prediction)
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
