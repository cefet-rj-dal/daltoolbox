#'@title Variational Autoencoder - Encode
#'@description Creates an deep learning variational autoencoder to encode a sequence of observations.
#' It wraps the pytorch library.
#'@param input_size input size
#'@param encoding_size encoding size
#'@param mean_var_size mean variable size
#'@param batch_size size for batch learning
#'@param num_epochs number of epochs for training
#'@param learning_rate learning rate
#'@return returns a `varae_encode_decode` object.
#'@examples
#'#See an example of using `varae_encode_decode` at this
#'#[link](https://github.com/cefet-rj-dal/daltoolbox/blob/main/transf/varae_enc_decode.ipynb)
#'@import reticulate
#'@export
varae_encode_decode <- function(input_size, encoding_size, mean_var_size=6, batch_size = 32, num_epochs = 1000, learning_rate = 0.001) {
  obj <- dal_transform()
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$mean_var_size <- mean_var_size
  obj$batch_size <- batch_size
  obj$num_epochs <- num_epochs
  obj$learning_rate <- learning_rate
  class(obj) <- append("varae_encode_decode", class(obj))

  return(obj)
}

#'@export
fit.varae_encode_decode <- function(obj, data, return_loss=FALSE, ...) {
  if (!exists("vae_create"))
    reticulate::source_python(system.file("python", "varae_autoencoder.py", package = "daltoolbox"))

  if (is.null(obj$model))
    obj$model <- vae_create(obj$input_size, obj$encoding_size, obj$mean_var_size)

  if (return_loss){
    fit_output <- vae_fit(obj$model, data, num_epochs = obj$num_epochs, learning_rate = obj$learning_rate, return_loss=return_loss)
    obj$model <- fit_output[[1]]

    return(list(obj=obj, loss=fit_output[-1]))
  }else{
    obj$model <- vae_fit(obj$model, data, num_epochs = obj$num_epochs, learning_rate = obj$learning_rate, return_loss=return_loss)
    return(obj)
  }
}

#'@export
transform.varae_encode_decode <- function(obj, data, ...) {
  if (!exists("vae_create"))
    reticulate::source_python(system.file("python", "varae_autoencoder.py", package = "daltoolbox"))

  result <- NULL
  if (!is.null(obj$model))
    result <- var_encode_decode(obj$model, data)
  return(result)
}
