source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

small <- iris[1:12, ]
small$Sepal.Length[c(3, 8)] <- NA
small$Sepal.Width[5] <- NA
small

small_complete <- na.omit(small)
small_complete

imp <- imputation_simple()
set_example_seed()
imp <- fit(imp, small)
small_imputed <- transform(imp, small)
small_imputed

out <- outliers_boxplot()
set_example_seed()
out <- fit(out, iris)
outliers_found <- transform(out, iris)
head(outliers_found)

cat_data <- data.frame(
  color = factor(c("red", "blue", "green", "red")),
  value = c(10, 20, 15, 12)
)

mapper <- categ_mapping("color")
cat_encoded <- transform(mapper, cat_data)
cat_encoded
