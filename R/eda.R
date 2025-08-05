library(tidyverse)
library(fs)

classification <- read_csv(fs::path(
  "data",
  "models",
  "20250802-120929",
  "ge2015_icecamp2.csv"
)) |>
  filter(probability >= 0.5)

classification |>
  count(predicted_class, sort = TRUE) |>
  print(n = 45L)

phyto_classes <- classification |>
  nest_by(predicted_class, .keep = TRUE)

destfolder <- fs::path("img", "classified_img")

if (dir_exists(destfolder)) {
  dir_delete(destfolder)
}

dir_create(destfolder)
dir_create(path(destfolder, unique(phyto_classes[["predicted_class"]])))

walk(phyto_classes$data, \(df) {
  file_copy(
    df$image_path,
    path(destfolder, df$predicted_class, path_file(df$image_path)),
    overwrite = TRUE
  )
})
