#'@title Convolutional 2d Denoising Autoencoder - Encode
#'@description Creates an deep learning convolutional denoising autoencoder to encode a sequence of observations.
#' It wraps the pytorch library.
#'@param input_size input size
#'@param encoding_size encoding size
#'@param batch_size size for batch learning
#'@param num_epochs number of epochs for training
#'@param learning_rate learning rate
#'@return a `c2den_encode_decode` object.
#'@examples
#'#See example at https://nbviewer.org/github/cefet-rj-dal/daltoolbox-examples
#'@import reticulate
#'@export
cae2den_encode_decode <- function(input_size, encoding_size, batch_size = 32, num_epochs = 50, learning_rate = 0.001) {
  obj <- dal_transform()
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$batch_size <- batch_size
  obj$num_epochs <- num_epochs
  print(num_epochs)
  obj$learning_rate <- learning_rate
  class(obj) <- append("cae2den_encode_decode", class(obj))

  return(obj)
}

#'@export
fit.cae2den_encode_decode <- function(obj, data, return_loss=FALSE, ...) {
  if (!exists("c2den_create"))
    reticulate::source_python(system.file("python", "conv2den_autoencoder.py", package = "daltoolbox"))

  if (is.null(obj$model))
    obj$model <- c2den_create(obj$input_size, obj$encoding_size)

  obj$input_size <- np_array(obj$input_size)
  
  if (return_loss){
    fit_output <- c2den_fit(obj$model, np_array(data), num_epochs = obj$num_epochs, learning_rate = obj$learning_rate, return_loss=return_loss)
    obj$model <- fit_output[[1]]
    
    return(list(obj=obj, loss=fit_output[-1]))
  }else{
    obj$model <- c2den_fit(obj$model, np_array(data), num_epochs = obj$num_epochs, learning_rate = obj$learning_rate)
    return(obj) 
  }
  
  

  return(obj)
}

#'@export
transform.cae2den_encode_decode <- function(obj, data, ...) {
  if (!exists("cae2den_create"))
    reticulate::source_python(system.file("python", "conv2den_autoencoder.py", package = "daltoolbox"))

  result <- NULL
  if (!is.null(obj$model))
    result <- c2den_encode_decode(obj$model, data)
  return(result)
}
