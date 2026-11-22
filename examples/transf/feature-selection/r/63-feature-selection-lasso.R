source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# installation
# install.packages("daltoolbox")

library(daltoolbox)

data(Diabetes)
head(Diabetes)

fs <- feature_selection_lasso("target")
set_example_seed()
fs <- fit(fs, Diabetes)

print(fs$selected)
print(fs$ranking)

diabetes_fs <- transform(fs, Diabetes)
head(diabetes_fs)
