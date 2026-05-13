source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "igraph"))

library(daltoolbox)

if (requireNamespace("igraph", quietly = TRUE)) {
  set_example_seed()
  g <- igraph::sample_gnp(n = 20, p = 0.15)
  g
}

if (requireNamespace("igraph", quietly = TRUE)) {
  model <- cluster_louvain_graph()
}

if (requireNamespace("igraph", quietly = TRUE)) {
  model <- fit(model, g)
  clu <- cluster(model, g)
  table(clu)
}

if (requireNamespace("igraph", quietly = TRUE)) {
  attr(clu, "modularity")
}

if (requireNamespace("igraph", quietly = TRUE)) {
  plot(
    g,
    vertex.color = as.factor(clu),
    vertex.label = NA,
    main = "Louvain communities"
  )
}
