import numpy as np
import pytest
from mrobotics.piecewise.waypoints_maker import make_rabbit_pattern # for the fixtures
from mrobotics.piecewise.cubic import cubic_interpolating_loop
from itertools import product

@pytest.fixture
def make_rabbit_curve():
    xy_waypts = make_rabbit_pattern()
    curve_under_test = cubic_interpolating_loop(xy_waypts)
    return curve_under_test

@pytest.mark.parametrize("method_name,periodicity",product(("pos","tang","utang","deri_tang", "tang_and_curv"),(-1,1,2,3,8, 12)))
def test_periodicity_eval(make_rabbit_curve, method_name, periodicity):
    curve_under_test = make_rabbit_curve#()
    s_test = np.linspace(curve_under_test.s_min, curve_under_test.s_max,30)
    s_test_another = s_test + int(periodicity)*curve_under_test.tot_dist

    if method_name != "tang_and_curv":
        expect_res = getattr(curve_under_test, "get_"+method_name)(s_test)
        actual_res = getattr(curve_under_test, "get_"+method_name)(s_test_another)
        np.testing.assert_allclose(actual_res, expect_res)
    else:
        expected_tang, expected_curv = curve_under_test.get_tang_and_curv(s_test)
        actual_tang,   actual_curv  = curve_under_test.get_tang_and_curv(s_test_another)
        np.testing.assert_allclose(actual_tang, expected_tang)
        np.testing.assert_allclose(actual_curv, expected_curv)

@pytest.mark.parametrize("xy_query,s_guess",[([4.0,5.0], 26.0), ([4.0,6.0], 26.0)])
def test_periodicity_projection(make_rabbit_curve,xy_query, s_guess):
    curve_under_test = make_rabbit_curve

    project_params = {
        "iter_max": 8,
        "soln_tolerance": 0.002# meter
    }
    # s stands for the curve parameter, which shall be (at least approximately) the arc-length parameterization
    # y stands for cross-track error
    expect_s, expect_y = curve_under_test.project(xy_query, s_guess, **project_params) 

    solns = [curve_under_test.project(xy_query, s_guess+k*curve_under_test.tot_dist, **project_params) for k in (-1 ,1, 2, 4, 10)]
    solns = np.array(solns)
    solns_s = solns[:,0]
    solns_y = solns[:,1]

    np.testing.assert_allclose(solns_y, expect_y)
    np.testing.assert_allclose([curve_under_test.wrap(soln_s) for soln_s in solns_s], expect_s, atol=project_params["soln_tolerance"]*0.5)
    
