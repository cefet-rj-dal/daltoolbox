#'@title Adversarial Autoencoder - Encode
#'@description Creates an deep learning adversarial autoencoder to encode a sequence of observations.
#' It wraps the pytorch library.
#'@param input_size input size
#'@param encoding_size encoding size
#'@param batch_size size for batch learning
#'@param num_epochs number of epochs for training
#'@param learning_rate learning rate
#'@return a `aae_encode_decode` object.
#'@examples
#'#See an example of using `aae_encode_decode` at this
#'#[link](https://github.com/cefet-rj-dal/daltoolbox/blob/main/transf/aae_enc_decode.ipynb)
#'@import reticulate
#'@export
aae_encode_decode <- function(input_size, encoding_size, batch_size = 32, num_epochs = 1000, learning_rate = 0.001, return_loss = FALSE, verbose = FALSE) {
  obj <- dal_transform()
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$batch_size <- batch_size
  obj$num_epochs <- num_epochs
  obj$learning_rate <- learning_rate
  obj$return_loss <- return_loss
  obj$verbose <- verbose
  class(obj) <- append("aae_encode_decode", class(obj))

  return(obj)
}

#'@export
fit.aae_encode_decode <- function(obj, data, ...){
  if (!exists("aae_create"))
    reticulate::source_python(system.file("python", "adv_autoencoder.py", package = "daltoolbox"))

  if (is.null(obj$model))
    obj$model <- aae_create(obj$input_size, obj$encoding_size)

  if (obj$return_loss){
    fit_output <- aae_fit(obj$model, data, num_epochs = obj$num_epochs, learning_rate = obj$learning_rate, return_loss=obj$return_loss, verbose=obj$verbose)
    obj$model <- fit_output[[1]]
    obj$loss <- fit_output[-1]
  }else{
    obj$model <- aae_fit(obj$model, data, num_epochs = obj$num_epochs, learning_rate = obj$learning_rate, return_loss=obj$return_loss, verbose=obj$verbose)
  }
  return(obj)
}

#'@export
transform.aae_encode_decode <- function(obj, data, ...) {
  if (!exists("aae_create"))
    reticulate::source_python(system.file("python", "adv_autoencoder.py", package = "daltoolbox"))

  result <- NULL
  if (!is.null(obj$model))
    result <- adv_encode_decode(obj$model, data)
  return(result)
}
