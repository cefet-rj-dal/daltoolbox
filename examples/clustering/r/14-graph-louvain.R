source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "igraph"))

library(daltoolbox)
library(igraph)

set_example_seed()
g <- igraph::sample_gnp(n = 20, p = 0.15)
g

model <- cluster_louvain_graph()

model <- fit(model, g)
clu <- cluster(model, g)
table(clu)

attr(clu, "modularity")

plot(
  g,
  vertex.color = as.factor(clu),
  vertex.label = NA,
  main = "Louvain communities"
)
