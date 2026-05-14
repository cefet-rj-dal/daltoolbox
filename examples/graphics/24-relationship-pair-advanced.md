About the chart
- `plot_pair_adv`: advanced pair plot with manual class coloring.

Didactic goal: show a slightly richer multivariate inspection where the palette is controlled explicitly, which is useful for presentation-ready comparisons.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "GGally", "RColorBrewer"))

library(daltoolbox)
library(GGally)
library(RColorBrewer)
```


``` r
colors <- brewer.pal(3, "Set1")
grf <- plot_pair_adv(
  datasets::iris,
  cnames = colnames(datasets::iris)[1:4],
  title = "Iris advanced pair plot",
  clabel = "Species",
  colors = colors
)
suppressMessages(suppressWarnings(print(grf)))
```

![plot of chunk unnamed-chunk-2](fig/24-relationship-pair-advanced/unnamed-chunk-2-1.png)
