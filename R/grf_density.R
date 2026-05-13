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

