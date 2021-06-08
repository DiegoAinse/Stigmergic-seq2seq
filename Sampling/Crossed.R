Álgebra <- data.frame(
  "Categoría" = c("Algebra", "Algebra", "Algebra", "Algebra"),
  "Módulo" = c("linear_1d", "linear_2d", "sequence_next_term","sequence_nth_term"),
  "Módulo de extrapolación" = c("polynomial_roots_big", "polynomial_roots_big", "polynomial_roots_big", "polynomial_roots_big")
)

Aritmética <- data.frame(
  "Categoría" = c("Aritmética", "Aritmética", "Aritmética", "Aritmética", "Aritmética", "Aritmética"),
  "Módulo" = c("add_or_sub_in_base", "add_or_sub_in_base", "add_or_sub_in_base", "nearest_integer_root", "nearest_integer_root", "nearest_integer_root", "simplify_surd", "simplify_surd", "simplify_surd"),
  "Módulo de extrapolación" = c("add_or_sub_big", "add_or_sub_big", "add_or_sub_big", "add_sub_multiple", "add_sub_multiple", "add_sub_multiple", "div_big", "div_big", "div_big", "mixed_longer", "mixed_longer", "mixed_longer", "mul_big", "mul_big", "mul_big", "mul_div_multiple_longer", "mul_div_multiple_longer", "mul_div_multiple_longer")
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
  "Módulo de extrapolación" = c("place_value_big", "round_number_big", )
)

Probabilidad <- data.frame(
  "Categoría" = c("Probabilidad", "Probabilidad"),
  "Módulo" = c("swr_p_level_set", "swr_p_sequence"),
  "Módulo de extrapolación" = c("swr_p_level_set_more_samples", "swr_p_sequence_more_samples")
)

Mathematics_dataset <- list(Álgebra, Aritmética, Comparación, Medición, Números, Probabilidad)
Mathematics_dataset

Extracción_aleatoria_Álgebra <- sample(1:4, 1, replace= F)
Extracción_aleatoria_Aritmética <- sample(1:4, 1, replace= F)
Extracción_aleatoria_Comparación <- sample(1:3, 1, replace= F)
Extracción_aleatoria_Medición <- sample(1:1, 1, replace= F)
Extracción_aleatoria_Números <- sample(1:2, 1, replace= F)
Extracción_aleatoria_Probabilidad <- sample(1:2, 1, replace= F)

Muestreo_Álgebra <- as.data.frame(Álgebra[Extracción_aleatoria_Álgebra,])
Muestreo_Aritmética <- as.data.frame(Aritmética[Extracción_aleatoria_Aritmética,])
Muestreo_Comparación <- as.data.frame(Comparación[Extracción_aleatoria_Comparación,])
Muestreo_Medición <- as.data.frame(Medición[Extracción_aleatoria_Medición,])
Muestreo_Números <- as.data.frame(Números[Extracción_aleatoria_Números,])
Muestreo_Probabilidad <- as.data.frame(Probabilidad[Extracción_aleatoria_Probabilidad,])

Muestra <- rbind(Muestreo_Álgebra, Muestreo_Aritmética, Muestreo_Comparación, Muestreo_Medición, Muestreo_Números, Muestreo_Probabilidad)
Muestra
