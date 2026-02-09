#'@title Plot advanced scatter matrix
#'@description Scatter matrix with class coloring and manual palette application.
#'@param data data.frame
#'@param cnames column names to include
#'@param title optional title
#'@param clabel optional class label column name
#'@param colors optional vector of colors for classes
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'data(iris)
#'grf <- plot_pair_adv(iris, cnames = colnames(iris)[1:4], title = "Iris")
#'print(grf)
#'@export
plot_pair_adv <- function(data, cnames, title = NULL, clabel = NULL, colors = NULL) {
  if (!requireNamespace("GGally", quietly = TRUE)) {
    stop("plot_pair_adv requires the 'GGally' package. Install with install.packages('GGally').")
  }

  if (!is.null(clabel)) {
    data$clabel <- data[, clabel]
    cnames <- c(cnames, "clabel")
  }
  icol <- match(cnames, colnames(data))
  icol <- icol[!is.na(icol)]
  if (length(icol) == 0) {
    stop("plot_pair_adv: no valid columns in 'cnames'.")
  }

  if (!is.null(clabel)) {
    grf <- GGally::ggpairs(data, columns = icol, ggplot2::aes(colour = clabel, alpha = 0.4)) + ggplot2::theme_bw(base_size = 10)
    if (!is.null(colors)) {
      for (i in 1:grf$nrow) {
        for (j in 1:grf$ncol) {
          grf[i, j] <- grf[i, j] + ggplot2::scale_fill_manual(values = colors) + ggplot2::scale_color_manual(values = colors)
        }
      }
    }
  } else {
    grf <- GGally::ggpairs(data, columns = icol) + ggplot2::theme_bw(base_size = 10)
  }
  if (!is.null(title)) grf <- grf + ggplot2::ggtitle(title)
  grf
}
