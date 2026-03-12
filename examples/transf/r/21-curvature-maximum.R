# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

# for ploting
library(ggplot2)
library(dplyr)

# Maximum curvature
# If the curve is decreasing, use maximum curvature analysis. 
# It brings a trade-off between having lower x values (with not so low y values) and having higher x values (not having to much decrease in y values). 

x <- seq(from=1,to=10,by=0.5)
dat <- data.frame(x = x, value = -log(x), variable = as.factor("log"))
myfit <- fit_curvature_max()
res <- transform(myfit, dat$value)
head(res)

grf <- plot_scatter(dat, label_x = "dimensions", label_y = "cumulative variance", colors="black") + 
    theme(text = element_text(size=16))
plot(grf + geom_vline(xintercept = dat$x[res$x], linetype="dashed", color = "red", size=0.5))

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

# for plotting
library(ggplot2)
library(dplyr)

x <- seq(from=1,to=10,by=0.5)
dat <- data.frame(x = x, value = -log(x), variable = "log")
dat$variable <- as.factor(dat$variable)
grf <- plot_scatter(dat, label_x = "x", label_y = "y", colors="black") + 
    theme(text = element_text(size=16))
plot(grf)

myfit <- fit_curvature_max()
res <- transform(myfit, dat$value)
res

plot(grf + geom_vline(xintercept = res$x, linetype="dashed", color = "red", size=0.5))
