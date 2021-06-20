Interpolación <- data.frame(
  "Categoría de Interpolación" = c("Álgebra", "Álgebra", "Álgebra", "Álgebra", "Álgebra", "Aritmética", "Aritmética", "Aritmética", "Aritmética", "Aritmética", "Aritmética", "Aritmética", "Aritmética", "Aritmética", "Cálculo", "Comparación", "Comparación", "Comparación", "Comparación", "Medida", "Medida", "Números", "Números", "Números", "Números", "Números", "Números", "Números", "Números", "Números", "Polinomios", "Polinomios", "Polinomios", "Polinomios", "Polinomios", "Polinomios", "Polinomios", "Probabilidad", "Probabilidad"),
  "Módulo de Interpolación" = c("linear_1d", "linear_2d", "polynomial_roots", "sequence_next_term", "sequence_nth_term", "add_or_sub", "add_or_sub_in_base", "add_sub_multiple", "div", "mixed", "mul", "mul_div_multiple", "nearest_integer_root", "simplify_surd", "differentiate", "closest", "kth_biggest", "pair", "sort", "conversion", "time", "base_conversion", "div_remainder", "gcd", "is_factor", "is_prime", "lcm", " list_prime_factors", "place_value", "round_number", "add", "collect", "compose", "coefficient_named", "evaluate", "expand", "simplify_power", "swr_p_level_set", "swr_p_sequence")
)

Extrapolación <- data.frame(
  "Categoría de extrapolación" = c("Álgebra", "Aritmética", "Aritmética", "Aritmética", "Aritmética", "Aritmética", "Comparación", "Comparación", "Comparación", "Medición", "Números", "Números", "Probabilidad", "Probabilidad"),
  "Módulo de extrapolación" = c("polynomial_roots_big", "add_sub_multiple", "div_big", "mixed_longer", "mul_big", "mul_div_multiple_longer", "closest_more", "kth_biggest_more", "sort_more", "conversion", "round_number_big","place_value_big", "swr_p_level_set_more_samples", "swr_p_sequence_more_samples")
)


Módulo_de_Interpolación <- Interpolación[sample(nrow(Interpolación), 1),]
Módulo_de_Extrapolación <- Extrapolación[sample(nrow(Extrapolación), 1),]
Módulo_de_Interpolación
Módulo_de_Extrapolación

Módulo_de_entrenamiento <- cbind(Módulo_de_Interpolación, Módulo_de_Extrapolación)
Módulo_de_entrenamiento