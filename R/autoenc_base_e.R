#'@title Autoencoder base (encoder)
#'@description Base class for encoder‑only autoencoders. Intended to be subclassed by
#' concrete implementations that learn a lower‑dimensional latent representation.
#'@details This base does not train or transform by itself (identity). Implementations should
#' override `fit()` to learn parameters and `transform()` to output the encoded representation.
#'@param input_size dimensionality of the input vector
#'@param encoding_size dimensionality of the latent (encoded) vector
#'@return returns an `autoenc_base_e` object
#'@examples
#'# See an end‑to‑end example at:
#'# https://github.com/cefet-rj-dal/daltoolbox/blob/main/autoencoder/autoenc_base_e.md
#'
#'@references
#' Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the Dimensionality of Data with Neural Networks. Science.
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
  # base class has no training; specialized implementations override
  return(obj)
}

#'@exportS3Method transform autoenc_base_e
transform.autoenc_base_e <- function(obj, data, ...) {
  # identity by default; specialized encoders should output latent representation
  return(data)
}
