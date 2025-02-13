% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/trans_aae_encode.R
\name{aae_encode}
\alias{aae_encode}
\title{Adversarial Autoencoder - Encode}
\usage{
aae_encode(
  input_size,
  encoding_size,
  batch_size = 350,
  num_epochs = 1000,
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
a \code{aae_encode} object.
#See an example of using \code{aae_encode} at this
#https://github.com/cefet-rj-dal/daltoolbox/blob/main/transf/aae_encode.md
}
\description{
Creates an deep learning adversarial autoencoder to encode a sequence of observations.
It wraps the pytorch library.
}
