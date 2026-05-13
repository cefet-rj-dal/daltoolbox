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

