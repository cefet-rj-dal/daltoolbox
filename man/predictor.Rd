% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dal_predictor.R
\name{predictor}
\alias{predictor}
\title{DAL Predict}
\usage{
predictor()
}
\value{
returns a predictor object
}
\description{
Ancestor class for regression and classification
It provides basis for fit and predict methods.
Besides, action method proxies to predict.

An example of learner is a decision tree (cla_dtree)
}
\examples{
#See ?cla_dtree for a classification example using a decision tree
}
