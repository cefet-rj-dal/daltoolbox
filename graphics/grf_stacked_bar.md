
```r
# DAL ToolBox
# version 1.1.727



#loading DAL
library(daltoolbox) 
```


```r
library(ggplot2)
library(RColorBrewer)

#color palette
colors <- brewer.pal(4, 'Set1')

# setting the font size for all charts
font <- theme(text = element_text(size=16))
```


```r
#iris dataset for the example
head(iris)
```

```
##   Sepal.Length Sepal.Width Petal.Length Petal.Width Species
## 1          5.1         3.5          1.4         0.2  setosa
## 2          4.9         3.0          1.4         0.2  setosa
## 3          4.7         3.2          1.3         0.2  setosa
## 4          4.6         3.1          1.5         0.2  setosa
## 5          5.0         3.6          1.4         0.2  setosa
## 6          5.4         3.9          1.7         0.4  setosa
```


```r
library(dplyr)

data <- iris |> group_by(Species) |> summarize(Sepal.Length=mean(Sepal.Length), Sepal.Width=mean(Sepal.Width))
head(data)
```

```
## # A tibble: 3 × 3
##   Species    Sepal.Length Sepal.Width
##   <fct>             <dbl>       <dbl>
## 1 setosa             5.01        3.43
## 2 versicolor         5.94        2.77
## 3 virginica          6.59        2.97
```

## Stacked-bar

Stacked bar charts are used to organize data into groups for each category.

A stacked bar chart stacks bars that represent different groups on top of each other. The height of the resulting bar shows the combined result of the groups. 

More information: https://en.wikipedia.org/wiki/Bar_chart#Grouped_or_stacked


```r
grf <- plot_stackedbar(data, colors=colors[1:2]) + font
grf <- grf + theme(axis.text.x = element_text(angle=90, hjust=1))
plot(grf)
```

![plot of chunk unnamed-chunk-5](fig/grf_stacked_bar/unnamed-chunk-5-1.png)

