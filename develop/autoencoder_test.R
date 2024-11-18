# DAL ToolBox
# version 1.0.777

source("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/jupyter.R")

#loading DAL
library(devtools)
load_all("/home/lucas/daltoolbox_autoenc")
library(ggpubr)
set.seed(1)

# Dataset
#data(sin_data)

sw_size <- 5

ts <- read.csv('/home/lucas/datasets/timeseries/eeg_eye_state/processed/eeg_eye_state.csv')
ts[,'X'] <- NULL
ts[,'eyeDetection'] <- NULL

# Feature Selection
selected_features <- c('F3', 'F4', 'AF3', 'AF4')
ts <- ts[selected_features]

# Remove Outliers
outl <- outliers(alpha=1.5)
outl <- fit(outl, ts)
ts <- transform(outl, ts)

ts_head(ts)

# Normalization
preproc <- ts_norm_gminmax()
preproc <- fit(preproc, ts)
ts <- transform(preproc, ts)

ts[ts > 1] <- 1
ts[ts < 0] <- 0

ts_head(ts)

# Train Test Split
samp <- ts_sample(ts, test_size = as.integer(nrow(ts)*0.3))
train <- as.data.frame(samp$train)
test <- as.data.frame(samp$test)
features <- names(train)

# Create Autoencoder
auto <- cae_encode(length(ts), encoding_size=2, num_epochs=40)
ae_type <- 'encoder'

return_loss <- TRUE
if (return_loss){
  auto <- fit(auto, train)
  
  train_loss <- unlist(auto$model$train_loss)
  val_loss <- unlist(auto$model$val_loss)
  
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

# Testing Autoencoder
result <- transform(auto, test)

pred_data <- rbind(train, test)
rec_data <- as.data.frame(transform(auto, pred_data[, features]))


ts_df <- rbind(train, test)
ts_df$index <- as.numeric(rownames(ts_df))

if (ae_type == 'encoder'){
  output_features <- names(rec_data)
  plot_data <-cbind(ts_df, rec_data[output_features])
  
  plot_features <- c(features, output_features)
  plotList <- lapply(
    plot_features,
    function(key) {
      plt <- ggplot(plot_data, aes(x=index, y=eval(parse(text=key)))) +
        geom_line() +
        xlab('') +
        ylab(key) + 
        theme_classic()
      
      plt
    }
  )
  
  ggarrange(
    plotlist=plotList,
    align='v',
    ncol=1, nrow=length(plot_features))
}else{
  pred_plot_data <- rec_data
  names(pred_plot_data) <- features
  
  output_features <- lapply(
    features,
    function(key) {
      new_string <- paste0(key, "_rec")      
      
      new_string
    }
  )
  
  ts_df$pred <- 0
  pred_plot_data$test_sample <- ts_df$test_sample
  pred_plot_data$index <- as.numeric(rownames(pred_plot_data))
  rownames(pred_plot_data) <- rownames(ts_df)
  names(pred_plot_data) <- c(output_features, c('index'))
  
  plot_data <- cbind(pred_plot_data, ts)
  
  plot_features <- output_features
  plotList <- lapply(
    output_features,
    function(key) {
      plt <- ggplot(plot_data, aes(x=index)) +
        geom_line(aes(y=eval(parse(text=key)), , colour='Reconstructed')) +
        geom_line(aes(y=eval(parse(text=substr(key, 1, nchar(key)-4))), colour='Original')) +
        xlab('') +
        ylab(substr(key, 1, nchar(key)-4)) +
        theme_classic()
      
      plt
    }
  )
  
  print(paste('MSE test:', mean(unlist((test - result)^2))))
  result <- as.data.frame(result)
  names(result) <- names(test)
  r2 <- c()
  for (col in names(test)){
    r2_col <- cor(test[col], result[col])^2
    r2 <- append(r2, r2_col)
    print(paste('R2 test:', col, r2_col))
  }
  print(paste('R2 test mean:', mean(r2)))
  
  
  
  ggarrange(
    plotlist=plotList,
    align='v',
    ncol=1, nrow=length(features))
}
