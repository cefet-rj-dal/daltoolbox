% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/graphics.R
\name{plot_hist}
\alias{plot_hist}
\title{Plot histogram}
\usage{
plot_hist(data, label_x = "", label_y = "", color = "white", alpha = 0.25)
}
\arguments{
\item{data}{data.frame contain x, value, and variable}

\item{label_x}{x-axis label}

\item{label_y}{y-axis label}

\item{color}{color vector}

\item{alpha}{transparency level}
}
\value{
returns a ggplot2::ggplot graphic
}
\description{
This function generates a histogram from a specified data frame using ggplot2.
}
\examples{
grf <- plot_hist(iris |> dplyr::select(Sepal.Width), color=c("blue"))
plot(grf)
}
