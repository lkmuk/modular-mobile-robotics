import numpy as np
from numpy import pi

__all__ = ["wrap_angle","rot_mat_2d", "apply_SE2_to_pts"]


def wrap_angle(theta):
    """
    wrap theta to (-pi, pi]

    copied from the MSR online course 2020
    """
    while theta < -pi:
        theta = theta + 2 * pi
    while theta > pi:
        theta = theta - 2 * pi
    return theta

def rot_mat_2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c,-s],
        [s, c]
    ])

def apply_SE2_to_pts(xy, CCW_angle, x_shift, y_shift):
    """
    1. rotation --> 2. translation

    This can also be used for coordinate mapping
    let the 2D points `xy` are in frame {B}
    the last three inputs denotes the SE(2) pose of frame {B} w.r.t. {A},
    output the coordinates of points `xy` in frame {A}

    Args:
        xy (numpy array): dimension N x 2 where N is the number of points
        CCW_angle (float): in radian 
        x_shift (float): in the same unit as xy
        y_shift (float): in the same unit as xy
    """

    xy_new = xy@rot_mat_2d(-CCW_angle) # Rot_z(theta)^T = Rot_z(-theta)
    xy_new[:,0] += x_shift
    xy_new[:,1] += y_shift
    return xy_new

# -------------------------------------------
#   refrain from using them;
#   consider using the pose2 abstraction instead 
#   --- there is an "inversion" method which creates a corresponding pose object
# ---------------------------------------------
def invert_SE(CCW_angle, x_shift, y_shift):
    CCW_angle_inv = -CCW_angle
    x_shift_inv, y_shift_inv = - rot_mat_2d(CCW_angle_inv)@np.array([x_shift, y_shift])
    return CCW_angle_inv, x_shift_inv, y_shift_inv

def apply_SE2_inv_to_pts(xy, CCW_angle, x_shift, y_shift):
    """
    When interpreted as coordinate mapping:
    let the 2D points `xy` are in frame {A}
    the last three inputs denotes the SE(2) pose of frame {B} w.r.t. {A}

    output the coordinates of points `xy` in frame {B}
    """
    xy_new = xy - np.array([x_shift, y_shift])
    # xy_new = rot_mat_2d(-CCW_angle)@xy_new
    xy_new = xy_new@rot_mat_2d(CCW_angle)
    return xy_new