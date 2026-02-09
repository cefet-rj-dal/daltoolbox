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

