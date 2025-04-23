data(sin_data)

sw_size <- 5
ts <- ts_data(sin_data$y, sw_size)

ts_head(ts)


preproc <- ts_norm_gminmax()
preproc <- fit(preproc, ts)
ts <- transform(preproc, ts)

ts_head(ts)

samp <- ts_sample(ts, test_size = 10)
train <- as.data.frame(samp$train)
test <- as.data.frame(samp$test)


auto <- autoenc_stacked_ed(5, 3)

auto <- fit(auto, train)

fit_loss <- data.frame(x=1:length(auto$train_loss), train_loss=auto$train_loss,val_loss=auto$val_loss)

grf <- plot_series(fit_loss, colors=c('Blue','Orange'))

plot(grf)

print(head(test))


# parte 2

load("~/harbinger/develop/train.rdata")

auto <- autoenc_ed(3, 2)

auto <- fit(auto, train)



# fitting the model
model <- fit(model, dataset$serie)

## 'list' object has no attribute 'to_numpy'

# making detections
detection <- detect(model, dataset$serie)

## Error in UseMethod("detect"): no applicable method for 'detect' applied to an object of class "c('autoenc_ed', 'dal_transform', 'dal_base')"

# filtering detected events
print(detection |> dplyr::filter(event==TRUE))

## Error: object 'detection' not found

# evaluating the detections
evaluation <- evaluate(model, detection$event, dataset$event)
print(evaluation$confMatrix)

## NULL

# ploting the results
grf <- har_plot(model, dataset$serie, detection, dataset$event)

## Error: object 'detection' not found

plot(grf)

## Error: object 'grf' not found

# ploting the results
res <-  attr(detection, "res")

## Error: object 'detection' not found

plot(res)

## Error: object 'res' not found


