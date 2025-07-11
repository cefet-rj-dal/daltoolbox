% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/graphics.R
\name{plot_series}
\alias{plot_series}
\title{Plot series}
\usage{
plot_series(data, label_x = "", label_y = "", colors = NULL)
}
\arguments{
\item{data}{data.frame contain x, value, and variable}

\item{label_x}{x-axis label}

\item{label_y}{y-axis label}

\item{colors}{color vector}
}
\value{
returns a ggplot2::ggplot graphic
}
\description{
This function creates a time series plot using ggplot2.
}
\examples{
x <- seq(0, 10, 0.25)
data <- data.frame(x, sin=sin(x))
head(data)

grf <- plot_series(data, colors=c("red"))
plot(grf)
}
