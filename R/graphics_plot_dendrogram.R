#'@title Plot dendrogram
#'@description Dendrogram plot for an hclust or dendrogram object using ggplot2.
#'@details Converts a dendrogram into line segments and renders it with ggplot2.
#'@param hc an object of class `hclust` or `dendrogram`
#'@param labels logical; whether to draw leaf labels
#'@param label_size label text size
#'@param title optional plot title
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'data(iris)
#'hc <- hclust(dist(scale(iris[,1:4])), method = "ward.D2")
#'grf <- plot_dendrogram(hc)
#'plot(grf)
#'@export
plot_dendrogram <- function(hc, labels = TRUE, label_size = 3, title = NULL) {
  x <- y <- xend <- yend <- label <- NULL
  if (inherits(hc, "hclust")) {
    dend <- stats::as.dendrogram(hc)
  } else if (inherits(hc, "dendrogram")) {
    dend <- hc
  } else {
    stop("plot_dendrogram: 'hc' must be hclust or dendrogram.")
  }

  leaf_counter <- 0

  build <- function(node) {
    if (stats::is.leaf(node)) {
      leaf_counter <<- leaf_counter + 1
      x <- leaf_counter
      y <- 0
      labs <- data.frame(
        x = x,
        y = y,
        label = attr(node, "label"),
        stringsAsFactors = FALSE
      )
      return(list(x = x, y = y, segments = NULL, labels = labs))
    }

    children <- lapply(node, build)
    xs <- vapply(children, function(ch) ch$x, numeric(1))
    ys <- vapply(children, function(ch) ch$y, numeric(1))
    y <- attr(node, "height")
    x <- mean(xs)

    segs <- do.call(
      rbind,
      lapply(children, function(ch) {
        data.frame(
          x = ch$x, y = ch$y,
          xend = ch$x, yend = y
        )
      })
    )
    segs <- rbind(
      segs,
      data.frame(
        x = min(xs), y = y,
        xend = max(xs), yend = y
      )
    )

    labs <- do.call(rbind, lapply(children, function(ch) ch$labels))
    return(list(x = x, y = y, segments = segs, labels = labs))
  }

  built <- build(dend)
  segs <- built$segments
  labs <- built$labels

  grf <- ggplot2::ggplot(segs, ggplot2::aes(x = x, y = y, xend = xend, yend = yend)) +
    ggplot2::geom_segment(linewidth = 0.3) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      panel.grid = ggplot2::element_blank(),
      axis.text.y = ggplot2::element_blank(),
      axis.ticks.y = ggplot2::element_blank()
    ) +
    ggplot2::xlab(NULL) +
    ggplot2::ylab(NULL)

  if (labels) {
    grf <- grf + ggplot2::geom_text(
      data = labs,
      ggplot2::aes(x = x, y = y, label = label),
      angle = 90,
      hjust = 1,
      size = label_size
    )
  }

  if (!is.null(title)) grf <- grf + ggplot2::ggtitle(title)
  return(grf)
}
