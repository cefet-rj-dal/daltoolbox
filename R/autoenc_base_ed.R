#'@title Autoencoder - Encode-decode
#'@description Creates a base class for autoencoder.
#'@param input_size input size
#'@param encoding_size encoding size
#'@return returns a `autoenc_base_ed` object.
#'@examples
#'#See an example of using `autoenc_base_ed` at this
#'#https://github.com/cefet-rj-dal/daltoolbox/blob/main/autoencoder/autoenc_base_ed.md
#'@export
autoenc_base_ed <- function(input_size, encoding_size) {
  obj <- dal_transform()
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  class(obj) <- append("autoenc_base_ed", class(obj))
  return(obj)
}

#'@exportS3Method fit autoenc_base_ed
fit.autoenc_base_ed <- function(obj, data, ...) {
  return(obj)
}

#'@exportS3Method transform autoenc_base_ed
transform.autoenc_base_ed <- function(obj, data, ...) {
  return(data)
}
