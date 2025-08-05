library(tidyverse)

read_csv("data/models/20250729-120937/phaeo.csv") |>
  count(predicted_class, sort = TRUE) |>
  mutate(accuracy = n / sum(n))
