#'@title Convolutional Autoencoder - Encode
#'@description Creates an deep learning convolutional autoencoder to encode a sequence of observations.
#' It wraps the pytorch library.
#'@param input_size input size
#'@param encoding_size encoding size
#'@param batch_size size for batch learning
#'@param num_epochs number of epochs for training
#'@param learning_rate learning rate
#'@return a `cae_encode_decode` object.
#'@examples
#'#See example at https://nbviewer.org/github/cefet-rj-dal/daltoolbox-examples
#'@import reticulate
#'@export
cae_encode_decode <- function(input_size, encoding_size, batch_size = 32, num_epochs = 1000, learning_rate = 0.001) {
  obj <- dal_transform()
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$batch_size <- batch_size
  obj$num_epochs <- num_epochs
  obj$learning_rate <- learning_rate
  class(obj) <- append("cae_encode_decode", class(obj))

  return(obj)
}

#'@export
fit.cae_encode_decode <- function(obj, data, return_loss=FALSE, ...) {
  if (!exists("cae_create"))
    reticulate::source_python(system.file("python", "conv_autoencoder.py", package = "daltoolbox"))

  if (is.null(obj$model))
    obj$model <- cae_create(obj$input_size, obj$encoding_size)

  if (return_loss){
    fit_output <- cae_fit(obj$model, data, num_epochs = obj$num_epochs, learning_rate = obj$learning_rate, return_loss=return_loss)
    obj$model <- fit_output[[1]]
    
    return(list(obj=obj, loss=fit_output[-1]))
  }else{
    obj$model <- cae_fit(obj$model, data, num_epochs = obj$num_epochs, learning_rate = obj$learning_rate, return_loss=return_loss)
    return(obj) 
  }
}

#'@export
transform.cae_encode_decode <- function(obj, data, ...) {
  if (!exists("cae_create"))
    reticulate::source_python(system.file("python", "conv_autoencoder.py", package = "daltoolbox"))

  result <- NULL
  if (!is.null(obj$model))
    result <- conv_encode_decode(obj$model, data)
  return(result)
}
