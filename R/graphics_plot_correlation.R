#'@title Plot correlation
#'@description Correlation heatmap with optional labels and triangle filtering.
#'@details Computes a correlation matrix from numeric columns (or `vars`) and renders a ggplot2
#' heatmap with values annotated. Supports reordering by hierarchical clustering or alphabetically.
#'@param df data.frame with numeric columns
#'@param vars optional vector of column names to include
#'@param method correlation method: "pearson", "spearman", or "kendall"
#'@param use handling of missing values for `stats::cor`
#'@param triangle which triangle to show: "full", "upper", or "lower"
#'@param reorder reordering strategy: "none", "hclust", or "alphabetical"
#'@param digits number of digits for labels
#'@param label_size size of label text
#'@param tile_color border color for tiles
#'@param show_diag whether to show the diagonal
#'@param title optional plot title
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'data(iris)
#'grf <- plot_correlation(iris[,1:4])
#'plot(grf)
#'@export
plot_correlation <- function(df,
                             vars = NULL,
                             method = c("pearson", "spearman", "kendall"),
                             use = "pairwise.complete.obs",
                             triangle = c("full", "upper", "lower"),
                             reorder = c("none", "hclust", "alphabetical"),
                             digits = 2,
                             label_size = 3,
                             tile_color = "white",
                             show_diag = TRUE,
                             title = NULL) {
  method   <- match.arg(method)
  triangle <- match.arg(triangle)
  reorder  <- match.arg(reorder)

  Var1 <- Var2 <- value <- label <- NULL
  if (!is.null(vars)) {
    if (!all(vars %in% names(df))) {
      bad <- vars[!vars %in% names(df)]
      stop("Vars ausentes: ", paste(bad, collapse = ", "))
    }
    df2 <- df[, vars, drop = FALSE]
  } else {
    df2 <- df[, vapply(df, is.numeric, logical(1)), drop = FALSE]
  }

  if (ncol(df2) < 2) stop("Precisa de ao menos 2 colunas numericas.")

  corr <- stats::cor(df2, use = use, method = method)

  if (reorder == "alphabetical") {
    ord <- order(colnames(corr))
    corr <- corr[ord, ord, drop = FALSE]
  } else if (reorder == "hclust") {
    d <- stats::as.dist(1 - corr)
    ord <- stats::hclust(d, method = "complete")$order
    corr <- corr[ord, ord, drop = FALSE]
  }

  vars1 <- rownames(corr)
  vars2 <- colnames(corr)
  grid <- expand.grid(Var1 = vars1, Var2 = vars2, stringsAsFactors = FALSE)
  grid$value <- as.vector(corr)
  grid$Var1 <- factor(grid$Var1, levels = vars1)
  grid$Var2 <- factor(grid$Var2, levels = vars2)
  grid$i <- as.integer(grid$Var1)
  grid$j <- as.integer(grid$Var2)

  if (triangle == "upper") {
    grid <- grid[grid$j > grid$i | (show_diag & grid$j == grid$i), , drop = FALSE]
  } else if (triangle == "lower") {
    grid <- grid[grid$i > grid$j | (show_diag & grid$j == grid$i), , drop = FALSE]
  } else {
    if (!show_diag) grid <- grid[grid$i != grid$j, , drop = FALSE]
  }

  grid$label <- ifelse(is.na(grid$value), "", format(round(grid$value, digits), nsmall = digits))

  oob_squish <- function(x, range) {
    pmin(pmax(x, range[1]), range[2])
  }

  ggplot2::ggplot(grid, ggplot2::aes(x = Var1, y = Var2, fill = value)) +
    ggplot2::geom_tile(color = tile_color, linewidth = 0.3) +
    ggplot2::geom_text(ggplot2::aes(label = label), size = label_size) +
    ggplot2::scale_fill_gradient2(
      low = "#4575b4",
      mid = "white",
      high = "#d73027",
      midpoint = 0,
      limits = c(-1, 1),
      oob = oob_squish,
      name = paste0("Corr (", method, ")")
    ) +
    ggplot2::coord_fixed() +
    ggplot2::labs(title = title, x = NULL, y = NULL) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      panel.grid = ggplot2::element_blank(),
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
      plot.title = ggplot2::element_text(face = "bold")
    )
}
