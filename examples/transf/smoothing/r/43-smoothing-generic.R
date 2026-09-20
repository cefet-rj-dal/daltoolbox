source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

obj <- smoothing(n = c(2, 3, 4, 5))
class(obj) <- append("smoothing_inter", class(obj))

set_example_seed()
obj <- fit(obj, datasets::iris$Sepal.Length)
smooth_values <- transform(obj, datasets::iris$Sepal.Length)

head(smooth_values)
obj$n
obj$interval
