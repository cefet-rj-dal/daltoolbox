source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "ggplot2", "RColorBrewer", "dplyr"))

library(daltoolbox)
library(ggplot2)
library(RColorBrewer)
library(dplyr)

colors <- brewer.pal(4, "Set1")
font <- theme(text = element_text(size = 16))

iris <- datasets::iris

gr <- plot_scatter(
  iris |>
    dplyr::select(x = Sepal.Length, value = Sepal.Width, variable = Species),
  label_x = "Sepal.Length",
  label_y = "Sepal.Width",
  colors = colors[1:3]
) + font

plot(gr)

set_example_seed()
sr <- train_test(sample_stratified("Species"), iris)
slevels <- levels(iris$Species)

models <- list(
  majority = cla_majority("Species", slevels),
  tree = cla_dtree("Species", slevels)
)

report <- lapply(names(models), function(name) {
set_example_seed()
  fitted <- fit(models[[name]], sr$train)
  pred <- predict(fitted, sr$test)
  metrics <- evaluate(fitted, sr$test$Species, pred)$metrics
  data.frame(model = name, t(metrics), check.names = FALSE)
})

do.call(rbind, report)
