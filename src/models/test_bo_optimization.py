import os
from entmoot.space.space import Space
from entmoot.optimizer.gurobi_utils import get_core_gurobi_model
import pandas as pd
from src.models.blackbox_sm_fnn import *

# Load data
initial_data = pd.read_csv('../../data/surrogate_model/sm_train_data_scaled.csv')

X_train = initial_data.iloc[:,:-4]
X_train.replace(to_replace=-1, value=0, inplace=True
               )
y_train = initial_data.iloc[:,-4:]

y_train = pd.Series(decode_classes(y_train.to_numpy()).flatten())

X_train = X_train.values.tolist()
y_train = y_train.values.tolist()

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

core_model.addConstr(emb_0 == 1)
core_model.addConstr(emb_1 == 1)
core_model.addConstr(emb_2 == 1)
core_model.addConstr(emb_3 == 1)
core_model.addConstr(emb_4 == 1)
core_model.addConstr(emb_5 == 1)
core_model.addConstr(emb_6 == 1)
core_model.addConstr(emb_7 == 1)
core_model.addConstr(emb_8 == 1)
core_model.addConstr(emb_9 == 1)
core_model.addConstr(fill_r == 1)
core_model.addConstr(fill_g == 1)
core_model.addConstr(fill_b == 1)
core_model.addConstr(svg_fill_r == 1)
core_model.addConstr(svg_fill_g == 1)
core_model.addConstr(svg_fill_b == 1)
core_model.addConstr(diff_fill_r == 1)
core_model.addConstr(diff_fill_g == 1)
core_model.addConstr(diff_fill_b == 1)
core_model.addConstr(rel_height == 1)
core_model.addConstr(rel_width == 1)
core_model.addConstr(rel_x_position == 1)
core_model.addConstr(rel_y_position == 1)
core_model.addConstr(rel_x_position_to_animations == 1)
core_model.addConstr(rel_y_position_to_animations == 1)
core_model.addConstr(nr_paths_svg == 1)

core_model.update()
core_model

from entmoot.optimizer.entmoot_minimize import entmoot_minimize

# cont_var_dict contains all continuous variabl

# specify the model core in `acq_optimizer_kwargs`
res = entmoot_minimize(
    func,
    func.get_bounds(),
    n_calls=1,
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
print("hi")