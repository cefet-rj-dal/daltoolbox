#'@title Plot time series with predictions
#'@description Plot original series plus dashed lines for in‑sample adjustment and optional out‑of‑sample predictions.
#'@details `yadj` length defines the training segment; `ypred` (if provided) is appended after `yadj`.
#'@param x time index (numeric vector) or NULL to use 1:length(y)
#'@param y numeric time series
#'@param yadj fitted/adjusted values for the training window
#'@param ypred optional predicted values after the training window
#'@param label_x x‑axis title
#'@param label_y y‑axis title
#'@param color color for the original series
#'@param color_adjust color for the adjusted values (dashed)
#'@param color_prediction color for the predictions (dashed)
#'@return returns a ggplot2::ggplot graphic
#'@examples
#'x <- base::seq(0, 10, 0.25)
#'yvalues <- sin(x) + rnorm(41,0,0.1)
#'adjust <- sin(x[1:35])
#'prediction <- sin(x[36:41])
#'grf <- plot_ts_pred(y=yvalues, yadj=adjust, ypred=prediction)
#'plot(grf)
#'@export
#'@importFrom ggplot2 ggplot
#'@importFrom ggplot2 aes
#'@importFrom ggplot2 geom_point
#'@importFrom ggplot2 geom_line
#'@importFrom ggplot2 scale_fill_manual
#'@importFrom ggplot2 theme_bw
#'@importFrom ggplot2 theme
#'@importFrom ggplot2 xlab
#'@importFrom ggplot2 ylab
#'@importFrom ggplot2 element_blank
#'@importFrom reshape melt
plot_ts_pred <- function(x = NULL, y, yadj, ypred = NULL, label_x = "", label_y = "", color="black", color_adjust="blue", color_prediction="green") {
  y <- as.vector(y)
  if (is.null(x))
    x <- 1:length(y)
  y <- as.vector(y)
  yadj <- as.vector(yadj)
  ntrain <- length(yadj)
  yhat <- yadj
  ntest <- 0
  if (!is.null(ypred)) {
    ypred <- as.vector(ypred)
    yhat <- c(yhat, ypred)
    ntest <- length(ypred)
  }

  grf <- ggplot2::ggplot() +
    ggplot2::geom_point( ggplot2::aes(x = x, y = y), color = color) +
    ggplot2::geom_line( ggplot2::aes(x = x, y = y), color = color) +
    ggplot2::xlab(label_x) +
    ggplot2::ylab(label_y) +
    ggplot2::theme_bw(base_size = 10) +
    ggplot2::theme(panel.grid.major = ggplot2::element_blank()) +
    ggplot2::theme(panel.grid.minor = ggplot2::element_blank()) +
    ggplot2::theme(legend.title = ggplot2::element_blank()) +
    ggplot2::theme(legend.position = "bottom") +
    ggplot2::theme(legend.key = ggplot2::element_blank())

  grf <- grf + ggplot2::geom_line( ggplot2::aes(x = x[1:ntrain], y = yhat[1:ntrain]),
                                   color = color_adjust, linetype = "dashed")
  if (!is.null(ypred))
    grf <- grf +ggplot2::geom_line( ggplot2::aes(x = x[ntrain:(ntrain+ntest)], y = yhat[ntrain:(ntrain+ntest)]),
                                    color = color_prediction, linetype = "dashed")
  return(grf)
}
