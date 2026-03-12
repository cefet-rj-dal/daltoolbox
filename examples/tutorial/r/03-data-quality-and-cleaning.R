# install.packages("daltoolbox")

library(daltoolbox)

small <- iris[1:12, ]
small$Sepal.Length[c(3, 8)] <- NA
small$Sepal.Width[5] <- NA
small

small_complete <- na.omit(small)
small_complete

out <- outliers_boxplot(features = c("Sepal.Length"))
out <- fit(out, iris)
outliers_found <- transform(out, iris)
head(outliers_found)

cat_data <- data.frame(
  color = factor(c("red", "blue", "green", "red")),
  value = c(10, 20, 15, 12)
)

mapper <- categ_mapping(features = "color")
mapper <- fit(mapper, cat_data)
cat_encoded <- transform(mapper, cat_data)
cat_encoded
