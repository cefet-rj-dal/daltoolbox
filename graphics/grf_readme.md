#  ggplot2
The ggplot2 is one of the most powerful packages for plotting chars.
Although very powerful, the learning curve for ggplot2 is high, as it is established on a grammar of graphics (https://www.amazon.com/Grammar-Graphics-Statistics-Computing/dp/0387245448) approach.

# graphics.R

The graphic.R enables plotting charts encapsulating ggplot2.
It enables an easy startup while learning how to use ggplot2. 

The majority of functions require a data.frame with two attributes or more attributes. In most cases, the first attribute is associated with the x-axis. In contrast, the second is related to the y-axis.

# Library
The library $myGraphics.R$ is loaded using the source function. 


``` r
# loading DAL
library(daltoolbox) 

# The easiest way to get ggplot2 is to install the whole tidyverse:
# install.packages("tidyverse")
# Alternatively, install just ggplot2:
# install.packages("ggplot2")
# Use suppressPackageStartupMessages(source(filename)) to avoid warning messages
```

# Color palette

One thing very relevant while plotting charts is to preserve visual identity. 
For that, the color brewer is an excellent tool to set up colors for your graphics using appropriate colors.
More information: https://colorbrewer2.org

Take some time to look at how to use it in R: https://rdrr.io/cran/RColorBrewer/man/ColorBrewer.html.


``` r
library(RColorBrewer)
col_set <- brewer.pal(9, 'Set1')
colors <- col_set[1:4]
```


``` r
library(ggplot2)
# setting the font size for all charts
font <- theme(text = element_text(size=16))
```

