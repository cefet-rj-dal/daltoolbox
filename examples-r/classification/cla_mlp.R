# Classification using Multilayer Perceptron

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

iris <- datasets::iris
head(iris)

# extracting the levels for the dataset
slevels <- levels(iris$Species)
slevels

# Construindo amostras (treino e teste) por amostragem aleatória
set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)
iris_train <- sr$train
iris_test <- sr$test


tbl <- rbind(table(iris[,"Species"]), 
             table(iris_train[,"Species"]), 
             table(iris_test[,"Species"]))
rownames(tbl) <- c("dataset", "training", "test")
head(tbl)

# Treinamento do modelo
model <- cla_mlp("Species", slevels, size=3, decay=0.03)
model <- fit(model, iris_train)


# Verificando ajuste no treino
train_prediction <- predict(model, iris_train)

# Avaliação do modelo (treino)
iris_train_predictand <- adjust_class_label(iris_train[,"Species"])
train_eval <- evaluate(model, iris_train_predictand, train_prediction)
print(train_eval$metrics)

# Teste do modelo
test_prediction <- predict(model, iris_test)

iris_test_predictand <- adjust_class_label(iris_test[,"Species"])

# Avaliação no teste
 test_eval <- evaluate(model, iris_test_predictand, test_prediction)
print(test_eval$metrics)
