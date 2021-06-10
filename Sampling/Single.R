Álgebra <- data.frame(
  "Categoría" = c("Algebra"),
  "Módulo" = c("polynomial_roots"),
  "Módulo de extrapolación" = c("polynomial_roots_big")
)

Aritmética <- data.frame(
  "Categoría" = c("Aritmética", "Aritmética", "Aritmética", "Aritmética", "Aritmética", "Aritmética"),
  "Módulo" = c("add_or_sub", "add_sub_multiple", "div", "mixed", "mul", "mul_div_multiple"),
  "Módulo de extrapolación" = c("add_or_sub_big", "add_sub_multiple", "div_big", "mixed_longer", "mul_big", "mul_div_multiple_longer")
)

Comparación <- data.frame(
  "Categoría" = c("Comparación", "Comparación", "Comparación"),
  "Módulo" = c("closest", "kth_biggest", "sort"),
  "Módulo de extrapolación" = c("closest_more", "kth_biggest_more", "sort_more")
)

Medición <- data.frame(
  "Categoría" = c("Medición"),
  "Módulo" = c("conversion"),
  "Módulo de extrapolación" = c("conversion")
)

Números <- data.frame(
  "Categoría" = c("Números", "Números"),
  "Módulo" = c("place_value", "round_number"),
  "Módulo de extrapolación" = c("place_value_big", "round_number_big")
)

Probabilidad <- data.frame(
  "Categoría" = c("Probabilidad", "Probabilidad"),
  "Módulo" = c("swr_p_level_set", "swr_p_sequence"),
  "Módulo de extrapolación" = c("swr_p_level_set_more_samples", "swr_p_sequence_more_samples")
)

Mathematics_dataset <- rbind(Álgebra, Aritmética, Comparación, Medición, Números, Probabilidad)
Mathematics_dataset

Módulo_de_entrenamiento <- Mathematics_dataset[sample(nrow(Mathematics_dataset), 1),]
Módulo_de_entrenamiento