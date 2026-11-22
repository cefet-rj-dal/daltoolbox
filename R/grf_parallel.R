#'@title Plot parallel coordinates
#'@description Parallel coordinates plot for multivariate data, with lines colored by group.
#'@details Each selected variable becomes a parallel vertical axis and each observation a
#' polyline crossing them, so class separation and multivariate patterns stand out. Variables are
#' standardized per column (z-score) before plotting so axes on different scales are comparable.
#'@param data data.frame
#'@param columns numeric columns to include (indices or names)
#'@param group grouping column (index or name)
#'@param colors optional vector of colors for groups
#'@param title optional title
#'@return returns a ggplot2::ggplot graphic
#'@references
#' Inselberg, A. (1985). The plane with parallel coordinates. The Visual Computer, 1(2), 69-91.
#' Wickham, H. (2016). ggplot2: Elegant Graphics for Data Analysis. Springer.
#'@examples
#'data(iris)
#'grf <- plot_parallel(iris, columns = 1:4, group = 5)
#'plot(grf)
#'@importFrom ggplot2 ggplot
#'@importFrom ggplot2 aes
#'@importFrom ggplot2 geom_line
#'@importFrom ggplot2 theme_bw
#'@importFrom ggplot2 theme
#'@importFrom ggplot2 scale_color_manual
#'@importFrom ggplot2 xlab
#'@importFrom ggplot2 ylab
#'@importFrom ggplot2 element_blank
#'@importFrom ggplot2 ggtitle
#'@importFrom reshape melt
#'@export
plot_parallel <- function(data, columns, group, colors = NULL, title = NULL) {
  .id <- .group <- value <- variable <- NULL
  df <- as.data.frame(data)
  if (is.numeric(columns)) columns <- colnames(df)[columns]
  if (is.numeric(group)) group <- colnames(df)[group]

  # standardize each axis (z-score), mirroring the common parallel-coordinates default
  scaled <- as.data.frame(scale(df[, columns, drop = FALSE]))
  scaled$.id <- seq_len(nrow(scaled))
  scaled$.group <- df[[group]]

  long <- reshape::melt(scaled, id.vars = c(".id", ".group"), measure.vars = columns)
  long$variable <- factor(long$variable, levels = columns)

  grf <- ggplot2::ggplot(long, ggplot2::aes(x = variable, y = value, group = .id, colour = .group)) +
    ggplot2::geom_line(alpha = 0.5) +
    ggplot2::theme_bw(base_size = 10) +
    ggplot2::theme(legend.title = ggplot2::element_blank(), legend.position = "bottom") +
    ggplot2::xlab("") + ggplot2::ylab("")
  if (!is.null(colors)) {
    grf <- grf + ggplot2::scale_color_manual(values = colors)
  }
  if (!is.null(title)) grf <- grf + ggplot2::ggtitle(title)
  return(grf)
}
