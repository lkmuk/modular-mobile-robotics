import numpy as np
from time import time

import matplotlib.pyplot as plt


def test_projection(path_obj, projection_method, test_query_pt, expected_LHS, init_guess_list, **algo_kwargs):
    """ test helper
    """
    # input validation
    if expected_LHS is not None:
        assert isinstance(expected_LHS, bool)
    assert projection_method in ("project", "project2"), "Not recognized!"
    assert projection_method in path_obj.__dir__(), f"Your input path object is of type {path_obj.__class__}, which does not have the requested projection method!"
    test_query_xy = np.array(test_query_pt).reshape(2) # btw validating the input

    res_st = np.zeros((len(init_guess_list),2),dtype=float) # Frenet frame coordinates of the projected points
    for i, init_guess in enumerate(init_guess_list):
        assert isinstance(init_guess,float), f"got {init_guess}..."
        t0 = time()
        res_arclength, res_distance = getattr(path_obj, projection_method)(test_query_pt ,arclength_init_guess=init_guess, **algo_kwargs)
        t1 = time()
        print(
            f"init guess: {init_guess:.2f}", f"final iterate: {res_arclength:.2f}",  
            f"(signed) projected distance: {res_distance:.2f}", 
            f"took {(t1-t0)*1e3:.2} msec", 
            sep=', '
        ) 
        print("-"*30)
        res_st[i] = [res_arclength, res_distance]

        if expected_LHS is not None:
            sign_expected = +1 if expected_LHS else -1
            assert res_distance*sign_expected >= 0.0, f"The projected distance (for initial guess {init_guess:.2f}) is of the wrong polarity!"
    return res_st

def viz_projection(path_obj, test_query_pt, init_guess_list, result_Frenet, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    assert len(init_guess_list) == len(result_Frenet)
    result_s = result_Frenet[:,0]

    path_obj.viz(ax=ax)
    for inital_guess_ind, s in enumerate(result_s):
        xy_projected = path_obj.get_pos(s).reshape(2)
        ax.plot(
            [test_query_pt[0], xy_projected[0]], 
            [test_query_pt[1], xy_projected[1]], 
            label=f'projection (init guess={init_guess_list[inital_guess_ind]:.2f})'
        )
    ax.plot(test_query_pt[0], test_query_pt[1], '^',label='query')
    ax.legend()