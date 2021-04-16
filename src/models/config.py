n_paths = 8
dim_path_vectors = 26
dim_animation_types = 6
dim_animation_parameters = 6
dim_animation_vector = dim_animation_types + dim_animation_parameters

# Animation predictor
a_input_size = 26
a_hidden_sizes = [15, 20]
a_out_sizes = [dim_animation_types, dim_animation_parameters]

# Surrogate model
s_hidden_sizes = [360,245]
