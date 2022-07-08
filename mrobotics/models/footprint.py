"""
Why bother with the footprint geometry of your robot?
* visualization (in top view) and/or
* collision detection

How to describe it?
* as a rectangle?
* as a circle/ ellipse?
* as a combination of circles

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from .. geom.basic_ops import apply_SE2_to_pts

__all__ = ["footprint_rectangle"]


class footprint_rectangle:
  def __init__(self, xf, xr, width):
    """ rectangular footprint geometry, for e.g. car-like robots
    Objects of this class provides (ordered) points that
    implies the boundaries of a planar robot's footprint.

    Here, we assume the footprint is a rectangle, specifically:

    * (ego-vehicle) coordinate system:
      xy is wrt the ego-vehicle frame, whose origin
      is attached to a reference point rigidly attached to the vehicle.
      The x-axis points forward while
      the y-axis points to the vehicle's left

    * Here, we assume the reference point to be on the 
      symmetry plane of the car-like vehicle
      (which simplifies the interfaces a bit).
      Possibility: Center of gravity (CG) or kinematic center
    """
    assert isinstance(xf, float) and xf > 0.01
    assert isinstance(xr, float) and xr > 0.01
    assert isinstance(width, float) and width > 0.01

    self.half_width = width/2
    self.xf = xf # front length extending from the reference point
    self.xr = xr # rear length relative to ...

    # do it only once 
    # (assuming your platform has a fixed footprint ...)
    self.xy_rect_box_pts = np.array([
        [self.xf, self.half_width], # front left corner
        [self.xf, -self.half_width], # front right corner
        [-self.xr, -self.half_width], # rear  right corner
        [-self.xr, +self.half_width] # rear left corner
    ])

  def get_rectangle_pts(self, X_ref,Y_ref, theta):
    return apply_SE2_to_pts(self.xy_rect_box_pts, theta, X_ref, Y_ref)
  
  def get_x_axis(self, X_ref, Y_ref, theta, length = None):
    """
    compute the floating vector that represents the platform's x-axis
    (in the world frame)
    The tail is the reference point.
    """
    if length is None:
      length = self.xf*0.7
    else:
      assert isinstance(length, float) and length > 0.0

    floating_unit_vec = np.array([ [0.,0.], [length, 0.0]]) # x-direction
    return apply_SE2_to_pts(
      floating_unit_vec, theta, X_ref, Y_ref
      )

  def draw_pose(self, X, Y, theta, ax, show_x_axis = True):
    if show_x_axis:
      ax.plot(
        *self.get_x_axis(X, Y, theta).T,
        '-r', marker='.',markevery=[0], markersize=8
      )
      
    ax.add_patch(
      Polygon(self.get_rectangle_pts(X, Y, theta),color='cyan',alpha=0.4)
    ) 
