% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/trans_c2den_encode.R, R/trans_cae2d_encode.R
\name{cae2den_encode}
\alias{cae2den_encode}
\title{Convolutional 2d Denoising Autoencoder - Encode}
\usage{
cae2den_encode(
  input_size,
  encoding_size,
  batch_size = 32,
  num_epochs = 50,
  learning_rate = 0.001
)

cae2den_encode(
  input_size,
  encoding_size,
  batch_size = 32,
  num_epochs = 50,
  learning_rate = 0.001
)
}
\arguments{
\item{input_size}{input size}

\item{encoding_size}{encoding size}

\item{batch_size}{size for batch learning}

\item{num_epochs}{number of epochs for training}

\item{learning_rate}{learning rate}
}
\value{
a \code{c2den_encode_decode} object.

a \code{cae2den_encode} object.
}
\description{
Creates an deep learning convolutional denoising autoencoder to encode a sequence of observations.
It wraps the pytorch library.

Creates an deep learning convolutional autoencoder to encode a sequence of observations.
It wraps the pytorch library.
}
\examples{
#See an example of using `c2den_encode_decode` at this
#https://github.com/cefet-rj-dal/daltoolbox/blob/main/transf/c2den_encode.md
#See an example of using `cae2den_encode` at this
#https://github.com/cefet-rj-dal/daltoolbox/blob/main/transf/cae2den_encode.md
}
