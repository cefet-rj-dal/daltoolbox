library(tidyverse)
#install.packages('torchvision')
library(torchvision)
#install.packages('imager')
library(imager)
library(caret)
library(stringr)
library(devtools)
load_all("/home/lucas/daltoolbox/")
options(scipen=999)
set.seed(1)

dataset_path <- '/home/lucas/datasets/images/kaggle_chest_c19_xrays/'
folder_list <- head(list.files(paste0(dataset_path, 'images/')), 3)
folder_paths <- paste0(dataset_path, 'images/', folder_list, '/images/')
file_names <- map(folder_paths, function(x) paste0(x, list.files(x))) %>%
  unlist()

# Print total number of images
print(length(file_names))

# Sample one for each class
class_examples <- c()
for (image_class in folder_list){
  class_images <- str_detect(file_names, image_class)
  class_example <- sample(file_names[class_images], 1)
  
  class_examples <- c(class_examples, class_example)
}

# Plot samples
img <- map(class_examples, load.image)

par(mfrow=(c(1,3)))
map(img, plot)

# Sample the data
sample_images <- sample(file_names, 400)

# Check dimensions
img <- load.image(file_names[1])
dim(img)

get_dim <- function(x){
  img <- load.image(x) 
  
  df_img <- data.frame(height = height(img),
                       width = width(img),
                       filename = x
  )
  
  return(df_img)
}

dim_df <- map_df(sample_images, get_dim)

input_size <- as.array(c(1, as.vector(unlist(dim_df[1, c('height', 'width')]))))

# Create Dataset
sample_size <- 200
train <- array(0, c(length(sample_images)/2, input_size[1], input_size[2], input_size[3]))
test <- array(0, c(length(sample_images)/2, input_size[1], input_size[2], input_size[3]))
a <- 1
for (i in sample_images[1:sample_size]){
  if (a <= sample_size){
    train[a, 1, , ] <- as.array(load.image(i))
  }else{
    test[a, 1, , ] <- as.array(load.image(i))
  }
  a <- a + 1
}

dim(train)
dim(test)

# Transform
auto <- cae2den_encode(input_size, encoding_size=4, num_epochs=20)
ae_type <- 'encoder'

return_loss <- FALSE
if (return_loss){
  fit_output <- fit(auto, train, return_loss=return_loss)
  auto <- fit_output[[1]]
  
  train_loss <- unlist(fit_output[['loss']][[1]])
  val_loss <- unlist(fit_output[['loss']][[2]])
  
  fit_loss <- as.data.frame(cbind(train_loss, val_loss))
  fit_loss['epoch'] <- 1:nrow(fit_loss)
  
  ggplot(fit_loss, aes(x=epoch)) +
    geom_line(aes(y=train_loss, colour='Train Loss')) +
    geom_line(aes(y=val_loss, colour='Val Loss')) +
    scale_color_manual(values=c('Blue','Orange')) +
    theme_classic()
}else{
  auto <- fit(auto, train, return_loss=return_loss)
}

result <- transform(auto, train)

dim(result)

if (ae_type == 'encoder'){
  ggplot(as.data.frame(result), aes(x=V1, y=V2)) +
    geom_point() +
    theme_classic()
  
}else{
  example_result <- result[1, 1, , ]
  
  dim(example_result)
  
  example_result[example_result > 1] <- 1
  
  rotate <- function(x) t(apply(x, 2, rev))
  
  plot_image <- function(x, col, img_list=plot_comparison){
    image(plot_comparison[[x]], col=col)
    title(x)
  }
  
  plot_comparison <- list()
  
  plot_comparison[['Input']] <- rotate(rotate(as.matrix(load.image(sample_images[length(sample_images)/2]))))
  plot_comparison[['Reconstructed']] <- rotate(rotate(as.matrix(example_result)))
  
  par(mfrow=(c(1,2)))
  map(names(plot_comparison), plot_image, col=grey.colors(50), img_list=plot_comparison)
}