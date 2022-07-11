"""Sample 2D points from some primitives/ commonly-used path

Here, each sequence of "waypoints" is a plain numpy arrays.
"""
import numpy as np

def gen_arc_points(xy_center, radius, start_radian, stop_radian, N_samples = 10):
    """generate waypoints along a circular arc

    inputs
    -----------------
    xy_center:    Nx2 numpy array
    radius:       +ve float
    start_radian: float
        corresponds to the first waypoint
    stop_radian:  float
        corresponds to the last waypoint
    N_samples:    +ve integer

    notes
    -----------------
    * The start and stop angles do NOT have to be bounded by [-pi,pi).
    * The arc (sequence) will be counter-clockwise (CCW)
      if stop_radian > start_radian. Otherwise, CW. 
    * The "arc" can have as many revolutions as you wish,
      (i.e. potentially spatially overlapping waypoints)

    """
    if stop_radian - start_radian > 0.0: #CCW
        angles = np.linspace(start_radian, stop_radian, N_samples)
        sign = +1
    else: #CW
        angles = np.linspace(stop_radian, start_radian, N_samples)#[::-1]
        sign = -1
    xy_arc = radius * np.vstack([np.cos(angles), np.sin(sign*angles)]).T
    return np.array(xy_center).reshape(1,2) + xy_arc