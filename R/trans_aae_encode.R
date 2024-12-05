#'@title Adversarial Autoencoder - Encode
#'@description Creates an deep learning adversarial autoencoder to encode a sequence of observations.
#' It wraps the pytorch library.
#'@param input_size input size
#'@param encoding_size encoding size
#'@param batch_size size for batch learning
#'@param num_epochs number of epochs for training
#'@param learning_rate learning rate
#'@return a `aae_encode` object.
#'#See an example of using `aae_encode` at this
#'#https://github.com/cefet-rj-dal/daltoolbox/blob/main/transf/aae_encode.md
#'@import reticulate
#'@export
aae_encode <- function(input_size, encoding_size, batch_size = 350, num_epochs = 1000, learning_rate = 0.001) {
  obj <- dal_transform()
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$batch_size <- batch_size
  obj$num_epochs <- num_epochs
  obj$learning_rate <- learning_rate
  class(obj) <- append("aae_encode", class(obj))

  return(obj)
}

#'@export
fit.aae_encode <- function(obj, data, ...) {
  if (!exists("aae_create"))
    reticulate::source_python(system.file("python", "adv_autoencoder.py", package = "daltoolbox"))

  if (is.null(obj$model))
    obj$model <- aae_create(obj$input_size, obj$encoding_size)
  
  obj$model <- aae_fit(obj$model, data, num_epochs = obj$num_epochs, learning_rate = obj$learning_rate)
  return(obj)
}

#'@export
transform.aae_encode <- function(obj, data, ...) {
  if (!exists("aae_create"))
    reticulate::source_python(system.file("python", "adv_autoencoder.py", package = "daltoolbox"))

  result <- NULL
  if (!is.null(obj$model))
    result <- adv_encode(obj$model, data)
  return(result)
}
