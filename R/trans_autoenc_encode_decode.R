#'@title Autoencoder - Encode-decode
#'@description Creates an deep learning autoencoder to encode-decode a sequence of observations.
#' It wraps the pytorch and reticulate libraries.
#'@param input_size input size
#'@param encoding_size encoding size
#'@param batch_size size for batch learning
#'@param num_epochs number of epochs for training
#'@param learning_rate learning rate
#'@return returns a `autoenc_encode_decode` object.
#'@examples
#'#See an example of using `autoenc_encode_decode` at this
#'#[link](https://github.com/cefet-rj-dal/daltoolbox/blob/main/transformation/enc_decode.ipynb)
#'@import reticulate
#'@export
autoenc_encode_decode <- function(input_size, encoding_size, batch_size = 32, num_epochs = 1000, learning_rate = 0.001) {
  obj <- dal_transform()
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$batch_size <- batch_size
  obj$num_epochs <- num_epochs
  obj$learning_rate <- learning_rate
  class(obj) <- append("autoenc_encode_decode", class(obj))

  return(obj)
}

#'@export
fit.autoenc_encode_decode <- function(obj, data, ...) {
  if (!exists("autoencoder_create"))
    reticulate::source_python(system.file("python", "autoencoder.py", package = "daltoolbox"))

  if (is.null(obj$model))
    obj$model <- autoencoder_create(obj$input_size, obj$encoding_size)

  if (return_loss){
    fit_output <- autoencoder_fit(obj$model, data, num_epochs = obj$num_epochs, learning_rate = obj$learning_rate, return_loss=return_loss)
    obj$model <- fit_output[[1]]
    
    return(list(obj=obj, loss=fit_output[-1]))
  }else{
    obj$model <- autoencoder_fit(obj$model, data, num_epochs = obj$num_epochs, learning_rate = obj$learning_rate, return_loss=return_loss)
    return(obj) 
  }
}

#'@export
transform.autoenc_encode_decode <- function(obj, data, ...) {
  if (!exists("autoencoder_create"))
    reticulate::source_python(system.file("python", "autoencoder.py", package = "daltoolbox"))

  result <- NULL
  if (!is.null(obj$model))
    result <- autoencoder_encode_decode(obj$model, data)
  return(result)
}
