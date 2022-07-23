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

def gen_s_curve(r1, r2):
    """
    Steering sequence along the path:
    1. straight driving
    2. 180-deg left turn (radius r1, in m)
    3. straight driving
    4. 180-deg right turn (radius r2, in m)
    5. straight driving

    return xy-coordinate of the waypoints (Nx2 numpy array)
    """
    assert r1 > 0.1
    assert r1 > 0.1
    return np.array([
        [0,0],
        [r1*0.50,0],
        [r1*0.75,0],
        *gen_arc_points([r1,r1],radius=r1, start_radian=-np.pi/2, stop_radian=np.pi/2, N_samples=6),
        [r1*0.75, 2*r1],
        [r1*0.55, 2*r1],
        [0, 2*r1],
        *gen_arc_points([-r2,2*r1+r2],radius=r2, start_radian=np.pi*1.5, stop_radian=np.pi/2, N_samples=6),
        [r1*0.55, 2*r1+2*r2],
        [r1, 2*r1+2*r2],
    ])


def make_rabbit_pattern():
    """
    motivation: a loop pattern with sharp corners
    """
    rhs = np.array([
        (2,-7),
        (7,-3),
        (7, 3),
        (4, 7.0),
        (7, 12.0),
        (8.0, 20.0),
        (7.5, 25.0),
        (4.0, 20.0),
        (1.0, 10.0),
    ])
    # first permutate in reverse order
    lhs = rhs[::-1].copy()
    # then reflect about the y-axis
    lhs[:,0] = -lhs[:,0]
    return np.vstack((rhs, lhs))