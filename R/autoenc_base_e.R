#'@title Autoencoder - Encode
#'@description Creates a base class for autoencoder.
#'@param input_size input size
#'@param encoding_size encoding size
#'@return returns a `autoenc_base_e` object.
#'@examples
#'#See an example of using `autoenc_base_e` at this
#'#https://github.com/cefet-rj-dal/daltoolbox/blob/main/autoencoder/autoenc_base_e.md
#'@export
autoenc_base_e <- function(input_size, encoding_size) {
  obj <- dal_transform()
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  class(obj) <- append("autoenc_base_e", class(obj))

  return(obj)
}

#'@exportS3Method fit autoenc_base_e
fit.autoenc_base_e <- function(obj, data, ...) {
  return(obj)
}

#'@exportS3Method transform autoenc_base_e
transform.autoenc_base_e <- function(obj, data, ...) {
  return(data)
}
