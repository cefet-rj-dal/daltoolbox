source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "GGally"))

library(daltoolbox)
library(GGally)

grf <- plot_parallel(datasets::iris, columns = 1:4, group = 5)
plot(grf)
