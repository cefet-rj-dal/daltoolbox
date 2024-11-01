data <- iris |> dplyr::group_by(Species) |>
  dplyr::summarize(Sepal.Length=mean(Sepal.Length))
head(data)

#ploting data
grf <- plot_bar(data, colors="blue")
plot(grf)
