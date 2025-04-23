#'@title Stacked Autoencoder - Encode
#'@description Creates an deep learning stacked autoencoder to encode a sequence of observations.
#'The autoencoder layers are based on DAL Toolbox Vanilla Autoencoder
#' It wraps the pytorch library.
#'@param input_size input size
#'@param encoding_size encoding size
#'@param batch_size size for batch learning
#'@param num_epochs number of epochs for training
#'@param learning_rate learning rate
#'@param k number of AE layers in the stack
#'@return a `autoenc_stacked_e_decode` object.
#'#See an example of using `autoenc_stacked_e_decode` at this
#'#https://github.com/cefet-rj-dal/daltoolbox/blob/main/autoencoder/autoenc_stacked_e.md
#'@import reticulate
#'@export
autoenc_stacked_e <- function(input_size, encoding_size, batch_size = 32, num_epochs = 1000, learning_rate = 0.001, k=3) {
  obj <- dal_transform()
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$batch_size <- batch_size
  obj$num_epochs <- num_epochs
  obj$learning_rate <- learning_rate
  obj$k <- k
  class(obj) <- append("autoenc_stacked_e", class(obj))

  return(obj)
}

#'@exportS3Method fit autoenc_stacked_e
fit.autoenc_stacked_e <- function(obj, data, ...) {
  if (!exists("autoenc_stacked_create"))
    reticulate::source_python(system.file("python", "autoenc_stacked.py", package = "daltoolbox"))

  if (is.null(obj$model))
    obj$model <- autoenc_stacked_create(obj$input_size, obj$encoding_size, obj$k)

  result <- autoenc_stacked_fit(obj$model, data, num_epochs = obj$num_epochs, learning_rate = obj$learning_rate)

  obj$model <- result[[1]]
  obj$train_loss <- result[[2]]
  obj$val_loss <- result[[3]]

  return(obj)
}

#'@exportS3Method transform autoenc_stacked_e
transform.autoenc_stacked_e <- function(obj, data, ...) {
  if (!exists("autoenc_stacked_create"))
    reticulate::source_python(system.file("python", "autoenc_stacked.py", package = "daltoolbox"))

  result <- NULL
  if (!is.null(obj$model)) {
    result <- autoenc_stacked_encode(obj$model, data)
  }
  return(result)
}
