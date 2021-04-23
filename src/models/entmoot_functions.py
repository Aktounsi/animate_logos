from entmoot.optimizer.optimizer import Optimizer

import copy
import inspect
import numbers
import pickle
import time
import pandas as pd
from src.models.blackbox_sm_fnn import *
from entmoot.space.space import Space
from entmoot.optimizer.gurobi_utils import get_core_gurobi_model

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


def entmoot_fit(
        dimensions,
        x0,
        y0,
        base_estimator="GBRT",
        std_estimator="BDD",
        n_initial_points=0,
        initial_point_generator="random",
        acq_func="LCB",
        acq_optimizer="global",
        random_state=None,
        acq_func_kwargs=None,
        acq_optimizer_kwargs=None,
        base_estimator_kwargs=None,
        std_estimator_kwargs=None,
        model_queue_size=None,
        verbose=1,
):

    specs = {"args": copy.copy(inspect.currentframe().f_locals),
             "function": inspect.currentframe().f_code.co_name}

    if acq_optimizer_kwargs is None:
        acq_optimizer_kwargs = {}

    # check x0: list-like, requirement of minimal points
    if x0 is None:
        x0 = []
    elif not isinstance(x0[0], (list, tuple)):
        x0 = [x0]
    if not isinstance(x0, list):
        raise ValueError("`x0` should be a list, but got %s" % type(x0))

    if n_initial_points <= 0 and not x0:
        raise ValueError("Either set `n_initial_points` > 0,"
                         " or provide `x0`")
    # check y0: list-like, requirement of maximal calls
    if isinstance(y0, Iterable):
        y0 = list(y0)
    elif isinstance(y0, numbers.Number):
        y0 = [y0]
    # calculate the total number of initial points
    n_initial_points = n_initial_points + len(x0)

    # Build optimizer

    # create optimizer class
    optimizer = Optimizer(
        dimensions,
        base_estimator=base_estimator,
        std_estimator=std_estimator,
        n_initial_points=n_initial_points,
        initial_point_generator=initial_point_generator,
        acq_func=acq_func,
        acq_optimizer=acq_optimizer,
        random_state=random_state,
        acq_func_kwargs=acq_func_kwargs,
        acq_optimizer_kwargs=acq_optimizer_kwargs,
        base_estimator_kwargs=base_estimator_kwargs,
        std_estimator_kwargs=std_estimator_kwargs,
        model_queue_size=model_queue_size,
        verbose=verbose
    )


    # record through tell function
    if x0:
        if not (isinstance(y0, Iterable) or isinstance(y0, numbers.Number)):
            raise ValueError(
                "`y0` should be an iterable or a scalar, got %s" % type(y0))
        if len(x0) != len(y0):
            raise ValueError("`x0` and `y0` should have the same length")
        result = optimizer.tell(x0, y0)
        result.specs = specs

    if isinstance(verbose, bool):
        if verbose:
            verbose = 1
        else:
            verbose = 0
    elif isinstance(verbose, int):
        if verbose not in [0, 1, 2]:
            raise TypeError("if verbose is int, it should in [0,1,2], "
                            "got {}".format(verbose))


    print('-'*60)
    print("Fitted a model to observed evaluations of the objective.")

    return optimizer


def entmoot_predict(optimizer, func, path_vector):
    specs = {"args": copy.copy(inspect.currentframe().f_locals),
             "function": inspect.currentframe().f_code.co_name}

    # initialize the search space manually
    space = Space(func.get_bounds())

    # get the core of the gurobi model from helper function 'get_core_gurobi_model'
    core_model = get_core_gurobi_model(space)

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

    core_model.addGenConstrIndicator(an_vec_0, True, an_vec_8 + an_vec_9 + an_vec_10 + an_vec_11 == 0)
    # core_model.addConstr(an_vec_8 == (1 - an_vec_0) * an_vec_8)
    # core_model.addConstr(an_vec_9 == (1 - an_vec_0) * an_vec_9)
    # core_model.addConstr(an_vec_10 == (1 - an_vec_0) * an_vec_10)
    # core_model.addConstr(an_vec_11 == (1 - an_vec_0) * an_vec_11)

    core_model.addGenConstrIndicator(an_vec_1, True, an_vec_6 + an_vec_7 + an_vec_9 + an_vec_10 + an_vec_11 == 0)
    # core_model.addConstr(an_vec_6 == (1 - an_vec_1) * an_vec_6)
    # core_model.addConstr(an_vec_7 == (1 - an_vec_1) * an_vec_7)
    # core_model.addConstr(an_vec_9 == (1 - an_vec_1) * an_vec_9)
    # core_model.addConstr(an_vec_10 == (1 - an_vec_1) * an_vec_10)
    # core_model.addConstr(an_vec_11 == (1 - an_vec_1) * an_vec_11)

    core_model.addGenConstrIndicator(an_vec_2, True, an_vec_6 + an_vec_7 + an_vec_8 + an_vec_10 + an_vec_11 == 0)
    # core_model.addConstr(an_vec_6 == (1 - an_vec_2) * an_vec_6)
    # core_model.addConstr(an_vec_7 == (1 - an_vec_2) * an_vec_7)
    # core_model.addConstr(an_vec_8 == (1 - an_vec_2) * an_vec_8)
    # core_model.addConstr(an_vec_10 == (1 - an_vec_2) * an_vec_10)
    # core_model.addConstr(an_vec_11 == (1 - an_vec_2) * an_vec_11)

    core_model.addGenConstrIndicator(an_vec_3, True, an_vec_6 + an_vec_7 + an_vec_8 + an_vec_9 == 0)
    # core_model.addConstr(an_vec_6 == (1 - an_vec_3) * an_vec_6)
    # core_model.addConstr(an_vec_7 == (1 - an_vec_3) * an_vec_7)
    # core_model.addConstr(an_vec_8 == (1 - an_vec_3) * an_vec_8)
    # core_model.addConstr(an_vec_9 == (1 - an_vec_3) * an_vec_9)

    core_model.addGenConstrIndicator(an_vec_4, True, an_vec_6 + an_vec_7 + an_vec_8 + an_vec_9 + an_vec_10 + an_vec_11 == 0)
    # core_model.addConstr(an_vec_6 == (1 - an_vec_4) * an_vec_6)
    # core_model.addConstr(an_vec_7 == (1 - an_vec_4) * an_vec_7)
    # core_model.addConstr(an_vec_8 == (1 - an_vec_4) * an_vec_8)
    # core_model.addConstr(an_vec_9 == (1 - an_vec_4) * an_vec_9)
    # core_model.addConstr(an_vec_10 == (1 - an_vec_4) * an_vec_10)
    # core_model.addConstr(an_vec_11 == (1 - an_vec_4) * an_vec_11)

    core_model.addGenConstrIndicator(an_vec_5, True, an_vec_6 + an_vec_7 + an_vec_8 + an_vec_9 + an_vec_10 + an_vec_11 == 0)
    # core_model.addConstr(an_vec_6 == (1 - an_vec_5) * an_vec_6)
    # core_model.addConstr(an_vec_7 == (1 - an_vec_5) * an_vec_7)
    # core_model.addConstr(an_vec_8 == (1 - an_vec_5) * an_vec_8)
    # core_model.addConstr(an_vec_9 == (1 - an_vec_5) * an_vec_9)
    # core_model.addConstr(an_vec_10 == (1 - an_vec_5) * an_vec_10)
    # core_model.addConstr(an_vec_11 == (1 - an_vec_5) * an_vec_11)

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

    optimizer.acq_optimizer_kwargs['add_model_core'] = core_model
    optimizer.update_next()
    next_x = optimizer.ask()
    next_y = func(next_x)

    #result = optimizer.tell(
    #     next_x, next_y,
    #     fit=False
    # )

    return next_x, next_y


if __name__ == '__main__':
    initial_data = pd.read_csv('../../data/surrogate_model/sm_train_data_scaled.csv')

    X_train = initial_data.iloc[:100, :-4]
    X_train.replace(to_replace=-1, value=0, inplace=True
                    )
    y_train = initial_data.iloc[:100, -4:]

    y_train = pd.Series(decode_classes(y_train.to_numpy()).flatten()) * -1

    X_train = X_train.values.tolist()
    y_train = y_train.values.tolist()

    # Load surrogate model
    func = SurrogateModelFNN()
    # initialize the search space manually
    start_time = time.time()
    optimizer = entmoot_fit(dimensions=func.get_bounds(), x0=X_train, y0=y_train, random_state=73)
    print("--- %s seconds ---" % (time.time() - start_time))
    with open('../../models/entmoot_optimizer.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(optimizer, output, pickle.HIGHEST_PROTOCOL)

# sample usage

    path_vector = [0.0000, 0.0000,
        0.0000, 0.0000,  0.8444,  0.7580, -2.1603,  0.4451, -0.3682,  0.2103,
         0.8382, -1.0178,  2.1813, -0.3529, -0.0390, -0.6908, -0.8948, -0.7838,
        -0.5913, -0.6961, -0.7385, -0.6718, -0.4203, -0.2536, -0.1224, -0.9218,
        -0.8612, -1.5663, -1.8816, -1.8314, -1.9453,  0.0352]
    opt_x, opt_y = entmoot_predict(optimizer, func, path_vector)