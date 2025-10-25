#' Boston Housing Data (Regression)
#' @description housing values in suburbs of Boston.
#' \itemize{
#' \item crim: per capita crime rate by town.
#' \item zn: proportion of residential land zoned for lots over 25,000 sq.ft.
#' \item indus: proportion of non-retail business acres per town
#' \item chas: Charles River dummy variable (= 1 if tract bounds)
#' \item nox: nitric oxides concentration (parts per 10 million)
#' \item rm: average number of rooms per dwelling
#' \item age: proportion of owner-occupied units built prior to 1940
#' \item dis: weighted distances to five Boston employment centres
#' \item rad: index of accessibility to radial highways
#' \item tax: full-value property-tax rate per $10,000
#' \item ptratio: pupil-teacher ratio by town
#' \item black: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#' \item lstat: percentage of lower status of the population
#' \item medv: Median value of owner-occupied homes in $1000's
#' }
#'
#' @docType data
#' @usage data(Boston)
#' @format Regression Dataset.
#' @keywords datasets
#' @references Creator: Harrison, D. and Rubinfeld, D.L.
#' Hedonic prices and the demand for clean air, J. Environ. Economics & Management, vol.5, 81-102, 1978.
#' @source This dataset was obtained from the MASS library.
#' @examples
#' data(Boston)
#' head(Boston)
# This file documents and registers the 'Boston' dataset for package users.
# The symbol below exposes the dataset when the package is loaded.
"Boston"

