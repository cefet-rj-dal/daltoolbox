source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

miner <- pattern_miner()
class(miner)
names(miner)
miner$pattern_kind
