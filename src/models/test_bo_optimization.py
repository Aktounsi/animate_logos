from entmoot.space.space import Space
from entmoot.optimizer.gurobi_utils import get_core_gurobi_model
import pandas as pd
from src.models.blackbox_sm_fnn import *
from entmoot.optimizer.entmoot_minimize import entmoot_minimize


# Load data
initial_data = pd.read_csv('../../data/surrogate_model/sm_train_data_scaled.csv')

X_train = initial_data.iloc[:200,:-4]
X_train.replace(to_replace=-1, value=0, inplace=True)
y_train = initial_data.iloc[:200,-4:]

y_train = pd.Series(decode_classes(y_train.to_numpy()).flatten()) * -1

X_train = X_train.values.tolist()
y_train = y_train.values.tolist()

path_vector = [0.4813, -1.0060, -0.5242,  1.3296,
         0.2993, -0.3018, -1.6347, -0.4806, -0.6526, -0.7260, -1.2620, -0.7265,
         0.3742,  0.2481,  0.4610,  0.6932, -1.5797, -1.1766, -0.1511,  1.6870,
        -0.3128,  0.0441,  0.8950,  0.9627,  0.0286, -0.5782]

# Load surrogate model
func = SurrogateModelFNN()
# initialize the search space manually
space = Space(func.get_bounds())

# get the core of the gurobi model from helper function 'get_core_gurobi_model'
core_model = get_core_gurobi_model(space)

# ordering of variable indices is dependent on space definition
an_vec_0 = core_model._cont_var_dict[0]
an_vec_1 = core_model._cont_var_dict[1]
an_vec_2 = core_model._cont_var_dict[2]
an_vec_3 = core_model._cont_var_dict[3]
an_vec_4 = core_model._cont_var_dict[4]
an_vec_5 = core_model._cont_var_dict[5]
an_vec_6 = core_model._cont_var_dict[6]
an_vec_7 = core_model._cont_var_dict[7]
an_vec_8 = core_model._cont_var_dict[8]
an_vec_9 = core_model._cont_var_dict[9]
an_vec_10 = core_model._cont_var_dict[10]
an_vec_11 = core_model._cont_var_dict[11]
emb_0 = core_model._cont_var_dict[12]
emb_1 = core_model._cont_var_dict[13]
emb_2 = core_model._cont_var_dict[14]
emb_3 = core_model._cont_var_dict[15]
emb_4 = core_model._cont_var_dict[16]
emb_5 = core_model._cont_var_dict[17]
emb_6 = core_model._cont_var_dict[18]
emb_7 = core_model._cont_var_dict[19]
emb_8 = core_model._cont_var_dict[20]
emb_9 = core_model._cont_var_dict[21]
fill_r = core_model._cont_var_dict[22]
fill_g = core_model._cont_var_dict[23]
fill_b = core_model._cont_var_dict[24]
svg_fill_r = core_model._cont_var_dict[25]
svg_fill_g = core_model._cont_var_dict[26]
svg_fill_b = core_model._cont_var_dict[27]
diff_fill_r = core_model._cont_var_dict[28]
diff_fill_g = core_model._cont_var_dict[29]
diff_fill_b = core_model._cont_var_dict[30]
rel_height = core_model._cont_var_dict[31]
rel_width = core_model._cont_var_dict[32]
rel_x_position = core_model._cont_var_dict[33]
rel_y_position = core_model._cont_var_dict[34]
rel_x_position_to_animations = core_model._cont_var_dict[35]
rel_y_position_to_animations = core_model._cont_var_dict[36]
nr_paths_svg = core_model._cont_var_dict[37]

core_model.addConstr(an_vec_0 + an_vec_1 + an_vec_2 + an_vec_3 + an_vec_4 + an_vec_5 == 1)

core_model.addConstr(an_vec_8 == (1-an_vec_0)*an_vec_8)
core_model.addConstr(an_vec_9 == (1-an_vec_0)*an_vec_9)
core_model.addConstr(an_vec_10 == (1-an_vec_0)*an_vec_10)
core_model.addConstr(an_vec_11 == (1-an_vec_0)*an_vec_11)

core_model.addConstr(an_vec_6 == (1 - an_vec_1) * an_vec_6)
core_model.addConstr(an_vec_7 == (1 - an_vec_1) * an_vec_7)
core_model.addConstr(an_vec_9 == (1 - an_vec_1) * an_vec_9)
core_model.addConstr(an_vec_10 == (1 - an_vec_1) * an_vec_10)
core_model.addConstr(an_vec_11 == (1 - an_vec_1) * an_vec_11)

core_model.addConstr(an_vec_6 == (1 - an_vec_2) * an_vec_6)
core_model.addConstr(an_vec_7 == (1 - an_vec_2) * an_vec_7)
core_model.addConstr(an_vec_8 == (1 - an_vec_2) * an_vec_8)
core_model.addConstr(an_vec_10 == (1 - an_vec_2) * an_vec_10)
core_model.addConstr(an_vec_11 == (1 - an_vec_2) * an_vec_11)

core_model.addConstr(an_vec_6 == (1 - an_vec_3) * an_vec_6)
core_model.addConstr(an_vec_7 == (1 - an_vec_3) * an_vec_7)
core_model.addConstr(an_vec_8 == (1 - an_vec_3) * an_vec_8)
core_model.addConstr(an_vec_9 == (1 - an_vec_3) * an_vec_9)

core_model.addConstr(an_vec_6 == (1 - an_vec_4) * an_vec_6)
core_model.addConstr(an_vec_7 == (1 - an_vec_4) * an_vec_7)
core_model.addConstr(an_vec_8 == (1 - an_vec_4) * an_vec_8)
core_model.addConstr(an_vec_9 == (1 - an_vec_4) * an_vec_9)
core_model.addConstr(an_vec_10 == (1 - an_vec_4) * an_vec_10)
core_model.addConstr(an_vec_11 == (1 - an_vec_4) * an_vec_11)

core_model.addConstr(an_vec_6 == (1 - an_vec_5) * an_vec_6)
core_model.addConstr(an_vec_7 == (1 - an_vec_5) * an_vec_7)
core_model.addConstr(an_vec_8 == (1 - an_vec_5) * an_vec_8)
core_model.addConstr(an_vec_9 == (1 - an_vec_5) * an_vec_9)
core_model.addConstr(an_vec_10 == (1 - an_vec_5) * an_vec_10)
core_model.addConstr(an_vec_11 == (1 - an_vec_5) * an_vec_11)

core_model.addConstr(emb_0 == path_vector[0])
core_model.addConstr(emb_1 == path_vector[1])
core_model.addConstr(emb_2 == path_vector[2])
core_model.addConstr(emb_3 == path_vector[3])
core_model.addConstr(emb_4 == path_vector[4])
core_model.addConstr(emb_5 == path_vector[5])
core_model.addConstr(emb_6 == path_vector[6])
core_model.addConstr(emb_7 == path_vector[7])
core_model.addConstr(emb_8 == path_vector[8])
core_model.addConstr(emb_9 == path_vector[9])
core_model.addConstr(fill_r == path_vector[10])
core_model.addConstr(fill_g == path_vector[11])
core_model.addConstr(fill_b == path_vector[12])
core_model.addConstr(svg_fill_r == path_vector[13])
core_model.addConstr(svg_fill_g == path_vector[14])
core_model.addConstr(svg_fill_b == path_vector[15])
core_model.addConstr(diff_fill_r == path_vector[16])
core_model.addConstr(diff_fill_g == path_vector[17])
core_model.addConstr(diff_fill_b == path_vector[18])
core_model.addConstr(rel_height == path_vector[19])
core_model.addConstr(rel_width == path_vector[20])
core_model.addConstr(rel_x_position == path_vector[21])
core_model.addConstr(rel_y_position == path_vector[22])
core_model.addConstr(rel_x_position_to_animations == path_vector[23])
core_model.addConstr(rel_y_position_to_animations == path_vector[24])
core_model.addConstr(nr_paths_svg == path_vector[25])

core_model.update()

# specify the model core in `acq_optimizer_kwargs`
res = entmoot_minimize(
    func,
    func.get_bounds(),
    n_calls=3,
    base_estimator="GBRT",
    std_estimator="L1DDP",
    n_initial_points=0,
    acq_func="LCB",
    acq_optimizer="global",
    x0=X_train,
    y0=y_train,
    random_state=100,
    acq_func_kwargs=None,
    acq_optimizer_kwargs={
      "add_model_core": core_model
    },
    std_estimator_kwargs=None,
    model_queue_size=None,
    base_estimator_kwargs={
        "min_child_samples": 2
    },
    verbose = True,
)

print(res['x_iters'][-1])

