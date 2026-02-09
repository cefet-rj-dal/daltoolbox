#'@title Plot parallel coordinates
#'@description Parallel coordinates plot using GGally::ggparcoord.
#'@param data data.frame
#'@param columns numeric columns to include (indices or names)
#'@param group grouping column (index or name)
#'@param colors optional vector of colors for groups
#'@param title optional title
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'data(iris)
#'grf <- plot_parallel(iris, columns = 1:4, group = 5)
#'plot(grf)
#'@export
plot_parallel <- function(data, columns, group, colors = NULL, title = NULL) {
  if (!requireNamespace("GGally", quietly = TRUE)) {
    stop("plot_parallel requer o pacote 'GGally'. Instale com install.packages('GGally').")
  }

  grf <- GGally::ggparcoord(data = data, columns = columns, group = group) +
    ggplot2::theme_bw(base_size = 10)
  if (!is.null(colors)) {
    grf <- grf + ggplot2::scale_color_manual(values = colors)
  }
  if (!is.null(title)) grf <- grf + ggplot2::ggtitle(title)
  grf
}
