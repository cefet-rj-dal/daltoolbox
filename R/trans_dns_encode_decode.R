#'@title Denoising Autoencoder - Encode
#'@description Creates an deep learning denoising autoencoder to encode a sequence of observations.
#' It wraps the pytorch library.
#'@param input_size input size
#'@param encoding_size encoding size
#'@param batch_size size for batch learning
#'@param num_epochs number of epochs for training
#'@param learning_rate learning rate
#'@param noise_factor level of noise to be added to the data
#'@return a `dns_encode_decode` object.
#'@examples
#'#See example at https://nbviewer.org/github/cefet-rj-dal/daltoolbox-examples
#'@import reticulate
#'@export
dns_encode_decode <- function(input_size, encoding_size, batch_size = 32, num_epochs = 1000, learning_rate = 0.001, noise_factor=0.3) {
  obj <- dal_transform()
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$batch_size <- batch_size
  obj$num_epochs <- num_epochs
  obj$learning_rate <- learning_rate
  obj$noise_factor <- noise_factor
  class(obj) <- append("dns_encode_decode", class(obj))
  
  return(obj)
}

#'@export
fit.dns_encode_decode <- function(obj, data, ...) {
  if (!exists("dns_ae_create"))
    reticulate::source_python(system.file("python", "dns_autoencoder.py", package = "daltoolbox"))
  
  if (is.null(obj$model))
    obj$model <- dns_ae_create(obj$input_size, obj$encoding_size, obj$noise_factor)
  
  obj$model <- dns_fit(obj$model, data, num_epochs = obj$num_epochs, learning_rate = obj$learning_rate)
  
  return(obj)
}

#'@export
transform.dns_encode_decode <- function(obj, data, ...) {
  if (!exists("dns_ae_create"))
    reticulate::source_python(system.file("python", "dns_autoencoder.py", package = "daltoolbox"))
  
  result <- NULL
  if (!is.null(obj$model))
    result <- dns_encode_decode(obj$model, data)
  return(result)
}
