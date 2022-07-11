"""Motion control algorithms for ground robots

Coding guidelines:
All algorithms ...
* should be state-less 
  (implement the stateful part in other wrappers/ functors) and 
* shall avoid unnecessary inputs/ dependence as much as possible
   e.g. 
   If an algorithm requires only the instantaneous reference curvature,
   just pass in the curvature;
   refrain from making the associated (planar) path object an input!
* BY DEFAULT, no checks on whether the assumptions are met (e.g. certain inputs to be +ve)

General references:
[MS08]: 
   P. Morin and C. Samson. 2008. 
   Ch.34 "Motion Control of Wheeled Mobile Robots" in Springer Handbook of Robotics
"""
from .. geom.basic_ops import wrap_angle
from .. models.motion.kinematic import v_front_axle_from_robot_state, calc_xy_front_axle # for Stanley controller
from . import cars
