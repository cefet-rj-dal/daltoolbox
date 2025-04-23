#'@title Denoising Autoencoder - Encode
#'@description Creates an deep learning denoising autoencoder to encode a sequence of observations.
#' It wraps the pytorch library.
#'@param input_size input size
#'@param encoding_size encoding size
#'@param batch_size size for batch learning
#'@param num_epochs number of epochs for training
#'@param learning_rate learning rate
#'@param noise_factor level of noise to be added to the data
#'@return a `autoenc_denoise_e_decode` object.
#'@examples
#'#See an example of using `autoenc_denoise_ed` at this
#'#https://github.com/cefet-rj-dal/daltoolbox/blob/main/autoencoder/autoenc_denoise_e.md
#'@import reticulate
autoenc_denoise_e <- function(input_size, encoding_size, batch_size = 32, num_epochs = 1000, learning_rate = 0.001, noise_factor=0.3) {
  obj <- dal_transform()
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$batch_size <- batch_size
  obj$num_epochs <- num_epochs
  obj$learning_rate <- learning_rate
  obj$noise_factor <- noise_factor
  class(obj) <- append("autoenc_denoise_e", class(obj))

  return(obj)
}

#'@exportS3Method fit autoenc_denoise_e
fit.autoenc_denoise_e <- function(obj, data, ...) {
  if (!exists("autoenc_denoise_create"))
    reticulate::source_python(system.file("python", "autoenc_denoise.py", package = "daltoolbox"))

  if (is.null(obj$model))
    obj$model <- autoenc_denoise_create(obj$input_size, obj$encoding_size, noise_factor = obj$noise_factor)

  result <- autoenc_denoise_fit(obj$model, data, num_epochs = obj$num_epochs, learning_rate = obj$learning_rate)

  obj$model <- result[[1]]
  obj$train_loss <- result[[2]]
  obj$val_loss <- result[[3]]


  return(obj)
}

#'@exportS3Method transform autoenc_denoise_e
transform.autoenc_denoise_e <- function(obj, data, ...) {
  if (!exists("autoenc_denoise_create"))
    reticulate::source_python(system.file("python", "autoenc_denoise.py", package = "daltoolbox"))

  result <- NULL
  if (!is.null(obj$model)) {
    result <- autoenc_denoise_encode(obj$model, data)
  }
  return(result)
}
