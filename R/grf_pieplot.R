#'@title Plot pie
#'@description Pie chart from a two‑column data.frame (category, value) using polar coordinates.
#'@details Slices are sized by the second (numeric) column. Text and border colors can be customized.
#'@param data two‑column data.frame with category and value
#'@param label_x x‑axis label (unused in pie, kept for symmetry)
#'@param label_y y‑axis label (unused in pie)
#'@param colors vector of slice fills
#'@param textcolor label text color
#'@param bordercolor slice border color
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'#summarizing iris dataset
#'data <- iris |> dplyr::group_by(Species) |>
#' dplyr::summarize(Sepal.Length=mean(Sepal.Length))
#'head(data)
#'
#'#ploting data
#'grf <- plot_pieplot(data, colors=c("red", "green", "blue"))
#'plot(grf)
#'@importFrom ggplot2 ggplot
#'@importFrom ggplot2 geom_bar
#'@importFrom ggplot2 aes
#'@importFrom ggplot2 coord_polar
#'@importFrom ggplot2 theme_light
#'@importFrom ggplot2 theme
#'@importFrom ggplot2 xlab
#'@importFrom ggplot2 ylab
#'@importFrom ggplot2 element_blank
#'@importFrom reshape melt
#'@importFrom dplyr filter
#'@importFrom dplyr group_by
#'@importFrom dplyr summarise
#'@importFrom dplyr arrange
#'@importFrom dplyr mutate
#'@export
plot_pieplot <- function(data, label_x = "", label_y = "", colors = NULL, textcolor="white", bordercolor="black") {
  x <- prop <- ypos <- label <- value <- desc <- n <- 0

  prepare.pieplot <- function(series) {
    colnames(series) <- c("x", "value")
    if (!is.factor(series$x)) {
      series$x <- as.factor(series$x)
    }

    series$colors <- colors

    series <- series |>
      dplyr::arrange(desc(x)) |>
      dplyr::mutate(prop = value / sum(series$value) *100) |>
      dplyr::mutate(ypos = cumsum(prop)- 0.5*prop) |>
      dplyr::mutate(label = paste(round(value / sum(value) * 100, 0), "%"))
    return(series)
  }
  series <- prepare.pieplot(data)

  # Basic piechart
  grf <- ggplot2::ggplot(series, ggplot2::aes(x="", y=prop, fill=x))
  grf <- grf + ggplot2::geom_bar(width = 1, stat = "identity", color=bordercolor)
  grf <- grf + ggplot2::theme_minimal(base_size = 10)
  grf <- grf + ggplot2::coord_polar("y", start=0)
  grf <- grf + ggplot2::geom_text( ggplot2::aes(y = ypos, label = label), size=6, color=textcolor)
  if (!is.null(colors))
    grf <- grf + ggplot2::scale_fill_manual(series$x, values = colors)
  grf <- grf + ggplot2::theme(panel.grid.minor = ggplot2::element_blank()) + ggplot2::theme(legend.position = "bottom")
  grf <- grf + ggplot2::xlab(label_x)
  grf <- grf + ggplot2::ylab(label_y)
  grf <- grf + ggplot2::theme(axis.text.x=ggplot2::element_blank(), legend.title = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank(), panel.grid = ggplot2::element_blank())
  return(grf)
}

