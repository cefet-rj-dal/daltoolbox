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
#'@return a `sae_encode_decode` object.
#'@examples
#'#See example at https://nbviewer.org/github/cefet-rj-dal/daltoolbox-examples
#'@import reticulate
#'@export
sae_encode_decode <- function(input_size, encoding_size, batch_size = 32, num_epochs = 1000, learning_rate = 0.001, k=3) {
  obj <- dal_transform()
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$batch_size <- batch_size
  obj$num_epochs <- num_epochs
  obj$learning_rate <- learning_rate
  obj$k <- k
  class(obj) <- append("sae_encode_decode", class(obj))
  
  return(obj)
}

#'@export
fit.sae_encode_decode <- function(obj, data, ...) {
  if (!exists("sae_create"))
    reticulate::source_python(system.file("python", "sae_autoencoder.py", package = "daltoolbox"))
  
  if (is.null(obj$model))
    obj$model <- sae_create(obj$input_size, obj$encoding_size, obj$k)
  
  obj$model <- sae_fit(obj$model, data, num_epochs = obj$num_epochs, learning_rate = obj$learning_rate)
  
  return(obj)
}

#'@export
transform.sae_encode_decode <- function(obj, data, ...) {
  if (!exists("sae_create"))
    reticulate::source_python(system.file("python", "sae_autoencoder.py", package = "daltoolbox"))
  
  result <- NULL
  if (!is.null(obj$model))
    result <- sae_encode_decode(obj$model, data)
  return(result)
}
