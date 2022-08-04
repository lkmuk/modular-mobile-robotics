import pytest
import numpy as np
from mrobotics.piecewise.polyline import polyline, polyline_from_bin


def test_read_write_waypt_file(tmp_path):
    xy = np.array([
        [0.34, -1.2],
        [30.1, 15.0],
        [20.0, 23.1]
    ])
    xy_obj = polyline(xy)
    xy_obj.save_as_bin(tmp_path/"waypt.data")

    xy_obj_retrieved = polyline_from_bin(tmp_path/"waypt.data")
    np.testing.assert_allclose(
        xy_obj_retrieved.XY_waypoints, 
        xy_obj.XY_waypoints, 
    )
    np.testing.assert_allclose(
        xy_obj_retrieved.idx2arclen, 
        xy_obj.idx2arclen, 
    )

def test_read_waypt_file_custom_bkpt(tmp_path):
    xy = np.array([
        [0.34, -1.2],
        [30.1, 15.0],
        [20.0, 23.1]
    ])
    xy_obj = polyline(xy)
    # some legitimate modification, e.g. offset
    # (they preserve moniticity)
    offset = 10.0
    xy_obj.idx2arclen = xy_obj.idx2arclen + offset 
    xy_obj.save_as_bin(tmp_path/"waypt_modified.data")
    print(xy_obj.idx2arclen)
    print(xy_obj.XY_waypoints)

    xy_recover1 = polyline_from_bin(tmp_path/"waypt_modified.data", keep_src_arc_length=False)
    np.testing.assert_allclose(
        xy_recover1.XY_waypoints, 
        xy_obj.XY_waypoints, 
    )
    np.testing.assert_allclose(
        xy_recover1.idx2arclen + offset, 
        xy_obj.idx2arclen , 
    )

    xy_recover2 = polyline_from_bin(tmp_path/"waypt_modified.data", keep_src_arc_length=True)
    np.testing.assert_allclose(
        xy_recover2.XY_waypoints, 
        xy_obj.XY_waypoints, 
    )
    np.testing.assert_allclose(
        xy_recover2.idx2arclen, 
        xy_obj.idx2arclen, 
    )