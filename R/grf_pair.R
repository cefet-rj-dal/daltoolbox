#'@title Plot scatter matrix
#'@description Scatter matrix using GGally::ggpairs with optional class coloring.
#'@param data data.frame
#'@param cnames column names to include
#'@param title optional title
#'@param clabel optional class label column name
#'@param colors optional vector of colors for classes
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'data(iris)
#'grf <- plot_pair(iris, cnames = colnames(iris)[1:4], title = "Iris")
#'print(grf)
#'@export
plot_pair <- function(data, cnames, title = NULL, clabel = NULL, colors = NULL) {
  if (!requireNamespace("GGally", quietly = TRUE)) {
    stop("plot_pair requires the 'GGally' package. Install with install.packages('GGally').")
  }

  plot_data <- as.data.frame(data)
  icol <- match(cnames, colnames(data))
  icol <- icol[!is.na(icol)]
  if (length(icol) == 0) {
    stop("plot_pair: no valid columns in 'cnames'.")
  }

  if (!is.null(clabel)) {
    if (!clabel %in% colnames(plot_data)) {
      stop("plot_pair: 'clabel' is not a valid column in 'data'.")
    }
    plot_data$.class_label <- plot_data[[clabel]]
    grf <- GGally::ggpairs(
      plot_data,
      columns = icol,
      mapping = ggplot2::aes(colour = .class_label, alpha = 0.4),
      progress = FALSE
    ) +
      ggplot2::theme_bw(base_size = 10)
    if (!is.null(colors)) {
      grf <- grf + ggplot2::scale_color_manual(values = colors)
    }
  } else {
    grf <- GGally::ggpairs(plot_data, columns = icol, progress = FALSE) + ggplot2::theme_bw(base_size = 10)
  }
  if (!is.null(title)) grf <- grf + ggplot2::ggtitle(title)
  grf
}
