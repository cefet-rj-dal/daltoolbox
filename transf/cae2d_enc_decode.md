## Autoencoder transformation (encode-decode)

Considering a dataset with $p$ numerical attributes. 

The goal of the autoencoder is to reduce the dimension of $p$ to $k$, such that these $k$ attributes are enough to recompose the original $p$ attributes. However from the $k$ dimensionals the data is returned back to $p$ dimensions. The higher the quality of autoencoder the similiar is the output from the input. 


```r
# DAL ToolBox
# version 1.1.727



library(tidyverse)
```

```
## ── Attaching core tidyverse packages ────────────────────────────────────────────────────────────────────────────────── tidyverse 2.0.0 ──
## ✔ dplyr     1.1.4     ✔ readr     2.1.5
## ✔ forcats   1.0.0     ✔ stringr   1.5.1
## ✔ lubridate 1.9.3     ✔ tibble    3.2.1
## ✔ purrr     1.0.2     ✔ tidyr     1.3.1
## ── Conflicts ──────────────────────────────────────────────────────────────────────────────────────────────────── tidyverse_conflicts() ──
## ✖ purrr::%||%()   masks base::%||%()
## ✖ dplyr::filter() masks stats::filter()
## ✖ dplyr::lag()    masks stats::lag()
## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors
```

```r
library(torchvision)
```

```
## Error in library(torchvision): there is no package called 'torchvision'
```

```r
library(imager)
```

```
## Error in library(imager): there is no package called 'imager'
```

```r
library(caret)
```

```
## Loading required package: lattice
## 
## Attaching package: 'caret'
## 
## The following object is masked from 'package:purrr':
## 
##     lift
## 
## The following object is masked from 'package:daltoolbox':
## 
##     cluster
```

```r
library(stringr)
library(devtools)
```

```
## Loading required package: usethis
```

```r
load_all("/home/lucas/daltoolbox_autoenc")
```

```
## Warning in normalizePath(path): path[1]="/home/lucas/daltoolbox_autoenc": No such file or directory
```

```
## Error in `value[[3L]]()`:
## ! Could not find a root 'DESCRIPTION' file that starts with '^Package' in '/home/lucas/daltoolbox_autoenc'.
## ℹ Are you in your project directory and does your project have a 'DESCRIPTION' file?
```

```r
#loading DAL
# library(daltoolbox)
```

### dataset for example 


```r
dataset_path <- '/home/lucas/datasets/images/kaggle_chest_c19_xrays/'
folder_list <- head(list.files(paste0(dataset_path, 'images/')), 3)
folder_paths <- paste0(dataset_path, 'images/', folder_list, '/images/')
file_names <- map(folder_paths, function(x) paste0(x, list.files(x))) %>%
  unlist()

# Print total number of images
print(length(file_names))
```

```
## [1] 1
```

### sampling one image for each class


```r
class_examples <- c()
for (image_class in folder_list){
  class_images <- str_detect(file_names, image_class)
  class_example <- sample(file_names[class_images], 1)
  
  class_examples <- c(class_examples, class_example)
}

# Plot samples
img <- map(class_examples, load.image)
```

```
## Error in eval(expr, envir, enclos): object 'load.image' not found
```

```r
par(mfrow=(c(1,3)))
p <- map(img, plot)
```

```
## Error in eval(expr, envir, enclos): object 'img' not found
```

### spliting into training and test


```r
# Sample the data
sample_images <- sample(file_names, 400)
```

```
## Error in sample.int(length(x), size, replace, prob): cannot take a sample larger than the population when 'replace = FALSE'
```

```r
# Check dimensions
img <- load.image(file_names[1])
```

```
## Error in load.image(file_names[1]): could not find function "load.image"
```

```r
dim(img)
```

```
## Error in eval(expr, envir, enclos): object 'img' not found
```

```r
get_dim <- function(x){
  img <- load.image(x) 
  
  df_img <- data.frame(height = height(img),
                       width = width(img),
                       filename = x
  )
  
  return(df_img)
}

dim_df <- map_df(sample_images, get_dim)
```

```
## Error in eval(expr, envir, enclos): object 'sample_images' not found
```

```r
input_size <- as.array(c(1, as.vector(unlist(dim_df[1, c('height', 'width')]))))
```

```
## Error in eval(expr, envir, enclos): object 'dim_df' not found
```

### spliting in train and test


```r
# Create Dataset
sample_size <- length(sample_images)/2
```

```
## Error in eval(expr, envir, enclos): object 'sample_images' not found
```

```r
train <- array(0, c(length(sample_images)/2, input_size[1], input_size[2], input_size[3]))
```

```
## Error in eval(expr, envir, enclos): object 'sample_images' not found
```

```r
test <- array(0, c(length(sample_images)/2, input_size[1], input_size[2], input_size[3]))
```

```
## Error in eval(expr, envir, enclos): object 'sample_images' not found
```

```r
a <- 1
for (i in sample_images[1:sample_size]){
  if (a <= sample_size){
    train[a, 1, , ] <- as.array(load.image(i))
  }else{
    test[a, 1, , ] <- as.array(load.image(i))
  }
  a <- a + 1
}
```

```
## Error in eval(expr, envir, enclos): object 'sample_images' not found
```

```r
dim(train)
```

```
## NULL
```

```r
dim(test)
```

```
## NULL
```

### creating autoencoder
Reduce from 5 to 3 dimensions


```r
auto <- cae2d_encode_decode(input_size, encoding_size=200, batch_size=100, num_epochs=250, learning_rate=0.01)
```

```
## Error in eval(expr, envir, enclos): object 'input_size' not found
```

```r
auto <- fit(auto, train)
```

```
## Error in eval(expr, envir, enclos): object 'auto' not found
```

### learning curves


```r
auto <- fit(auto, train)
```

```
## Error in eval(expr, envir, enclos): object 'auto' not found
```

```r
train_loss <- unlist(auto$model$train_loss)
```

```
## Error in eval(expr, envir, enclos): object 'auto' not found
```

```r
val_loss <- unlist(auto$model$val_loss)
```

```
## Error in eval(expr, envir, enclos): object 'auto' not found
```

```r
fit_loss <- as.data.frame(cbind(train_loss, val_loss))
```

```
## Error in eval(expr, envir, enclos): object 'train_loss' not found
```

```r
fit_loss['epoch'] <- 1:nrow(fit_loss)
```

```
## Error in eval(expr, envir, enclos): object 'fit_loss' not found
```

```r
ggplot(fit_loss, aes(x=epoch)) +
geom_line(aes(y=train_loss, colour='Train Loss')) +
geom_line(aes(y=val_loss, colour='Val Loss')) +
scale_color_manual(values=c('Blue','Orange')) +
theme_classic()
```

```
## Error in eval(expr, envir, enclos): object 'fit_loss' not found
```

### testing autoencoder
presenting the original test set and display encoding


```r
result <- transform(auto, train)
```

```
## Error in eval(expr, envir, enclos): object 'auto' not found
```

```r
example_result <- result[1, 1, , ]
```

```
## Error in eval(expr, envir, enclos): object 'result' not found
```

```r
dim(example_result)
```

```
## Error in eval(expr, envir, enclos): object 'example_result' not found
```

```r
example_result[example_result > 1] <- 1
```

```
## Error in eval(ei, envir): object 'example_result' not found
```

```r
rotate <- function(x) t(apply(x, 2, rev))

plot_image <- function(x, col, img_list=plot_comparison){
image(plot_comparison[[x]], col=col)
title(x)
}

plot_comparison <- list()

plot_comparison[['Input']] <- rotate(rotate(as.matrix(load.image(sample_images[length(sample_images)/2]))))
```

```
## Error in load.image(sample_images[length(sample_images)/2]): could not find function "load.image"
```

```r
plot_comparison[['Reconstructed']] <- rotate(rotate(as.matrix(example_result)))
```

```
## Error in eval(expr, envir, enclos): object 'example_result' not found
```

```r
par(mfrow=(c(1,2)))
map(names(plot_comparison), plot_image, col=grey.colors(50), img_list=plot_comparison)
```

```
## list()
```

