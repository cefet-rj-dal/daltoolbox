#'@title Plot pixel visualization
#'@description Pixel-oriented visualization of a numeric matrix or data.frame.
#'@details Renders a heatmap-like plot where each cell is a pixel. Useful for multivariate inspection.
#'@param data numeric matrix or data.frame
#'@param colors optional vector of colors for the fill gradient
#'@param title optional plot title
#'@param label_x x-axis label
#'@param label_y y-axis label
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'data(iris)
#'grf <- plot_pixel(as.matrix(iris[,1:4]), title = "Iris")
#'plot(grf)
#'@export
plot_pixel <- function(data, colors = NULL, title = NULL, label_x = "sample", label_y = "Attributes") {
  x <- y <- value <- NULL
  mat <- as.matrix(data)
  nrow_mat <- nrow(mat)
  ncol_mat <- ncol(mat)

  grid <- expand.grid(
    x = seq_len(nrow_mat),
    y = seq_len(ncol_mat)
  )
  grid$value <- as.vector(mat)

  grf <- ggplot2::ggplot(grid, ggplot2::aes(x = x, y = y, fill = value)) +
    ggplot2::geom_raster() +
    ggplot2::scale_y_continuous(breaks = seq_len(ncol_mat), labels = colnames(mat)) +
    ggplot2::scale_x_continuous(breaks = seq(1, nrow_mat, by = 10)) +
    ggplot2::labs(title = title, x = label_x, y = label_y) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(panel.grid = ggplot2::element_blank())

  if (!is.null(colors)) {
    grf <- grf + ggplot2::scale_fill_gradientn(colors = colors)
  } else {
    grf <- grf + ggplot2::scale_fill_gradient(low = "white", high = "steelblue")
  }

  return(grf)
}
