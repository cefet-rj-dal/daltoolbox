# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

# for ploting
library(ggplot2)
library(dplyr)

wine <- get(load(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/wine.RData")))
head(wine)

pca_res = prcomp(wine[,2:ncol(wine)], center=TRUE, scale.=TRUE)
y <- cumsum(pca_res$sdev^2/sum(pca_res$sdev^2)) # cumulative variance
x <- 1:length(y)

dat <- data.frame(x, value = y, variable = "PCA")
dat$variable <- as.factor(dat$variable)
head(dat)

grf <- plot_scatter(dat, label_x = "dimensions", label_y = "cumulative variance", colors="black") + 
    theme(text = element_text(size=16))
plot(grf)

myfit <- fit_curvature_min()
res <- transform(myfit, y)  # returns optimal index (knee)
head(res)

plot(grf + geom_vline(xintercept = res$x, linetype="dashed", color = "red", size=0.5))

