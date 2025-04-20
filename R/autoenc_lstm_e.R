#'@title LSTM Autoencoder - Encode
#'@description Creates an deep learning LSTM autoencoder to encode a sequence of observations.
#' It wraps the pytorch library.
#'@param input_size input size
#'@param encoding_size encoding size
#'@param batch_size size for batch learning
#'@param num_epochs number of epochs for training
#'@param learning_rate learning rate
#'@return returns a `autoenc_lstm_e` object.
#'@examples
#'#See an example of using `autoenc_lstm_e` at this
#'#https://github.com/cefet-rj-dal/daltoolbox/blob/main/autoencoder/autoenc_lstm_e.md
#'@import reticulate
#'@export
autoenc_lstm_e <- function(input_size, encoding_size, batch_size = 32, num_epochs = 50, learning_rate = 0.001) {
  obj <- dal_transform()
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$batch_size <- batch_size
  obj$num_epochs <- num_epochs
  obj$learning_rate <- learning_rate
  class(obj) <- append("autoenc_lstm_e", class(obj))

  return(obj)
}

#'@export
fit.autoenc_lstm_e <- function(obj, data, ...) {
  if (!exists("autoenc_lstm_create"))
    reticulate::source_python(system.file("python", "lstm_autoencoder.py", package = "daltoolbox"))

  if (is.null(obj$model))
    obj$model <- autoenc_lstm_create(obj$input_size, obj$encoding_size)

  result <- autoenc_lstm_fit(obj$model, data, num_epochs = obj$num_epochs, learning_rate = obj$learning_rate)

  obj$model <- result[[1]]
  obj$train_loss <- unlist(result[[2]]$tolist())
  obj$val_loss <- unlist(result[[3]]$tolist())

  return(obj)
}

#'@export
transform.autoenc_lstm_e <- function(obj, data, ...) {
  if (!exists("autoenc_lstm_create"))
    reticulate::source_python(system.file("python", "lstm_autoencoder.py", package = "daltoolbox"))

  result <- NULL
  if (!is.null(obj$model)) {
    result <- autoenc_lstm_encode(obj$model, data)
    result <- matrix(unlist(result$tolist()), ncol = obj$encoding_size, byrow = TRUE)
  }
  return(result)
}
