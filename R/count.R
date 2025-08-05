library(tidyverse)
library(fs)


dir_ls("./data/val/", recurse = TRUE, glob = "*.png") |>
  enframe() |>
  count(dirname(name), sort = TRUE)
