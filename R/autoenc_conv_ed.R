#'@title Convolutional Autoencoder - Encode
#'@description Creates an deep learning convolutional autoencoder to encode a sequence of observations.
#' It wraps the pytorch library.
#'@param input_size input size
#'@param encoding_size encoding size
#'@param batch_size size for batch learning
#'@param num_epochs number of epochs for training
#'@param learning_rate learning rate
#'@return a `autoenc_conv_ed` object.
#'@examples
#'#See an example of using `autoenc_conv_ed` at this
#'#https://github.com/cefet-rj-dal/daltoolbox/blob/main/transf/autoenc_conv_ed.md
#'@import reticulate
#'@export
autoenc_conv_ed <- function(input_size, encoding_size, batch_size = 32, num_epochs = 1000, learning_rate = 0.001) {
  obj <- dal_transform()
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$batch_size <- batch_size
  obj$num_epochs <- num_epochs
  obj$learning_rate <- learning_rate
  class(obj) <- append("autoenc_conv_ed", class(obj))

  return(obj)
}

#'@export
fit.autoenc_conv_ed <- function(obj, data, ...) {
  if (!exists("autoenc_conv_create"))
    reticulate::source_python(system.file("python", "autoenc_conv.py", package = "daltoolbox"))

  if (is.null(obj$model))
    obj$model <- autoenc_conv_create(obj$input_size, obj$encoding_size)

  result <- autoenc_conv_fit(obj$model, data, num_epochs = obj$num_epochs, learning_rate = obj$learning_rate)

  obj$model <- result[[1]]
  obj$train_loss <- result[[2]]
  obj$val_loss <- result[[3]]

  return(obj)
}

#'@export
transform.autoenc_conv_ed <- function(obj, data, ...) {
  if (!exists("autoenc_conv_create"))
    reticulate::source_python(system.file("python", "autoenc_conv.py", package = "daltoolbox"))

  result <- NULL
  if (!is.null(obj$model)) {
    result <- autoenc_conv_encode_decode(obj$model, data)
  }

  return(result)
}
