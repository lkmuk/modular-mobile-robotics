import numpy as np
# for the non-periodic case
from scipy.interpolate import splrep, splev, splder # splint is not useful in our case
# for the periodic case
from scipy.interpolate import CubicSpline

from .base import planar_curve_deg3 # the API + some functionalities
from .polyline import polyline # for the breakpoint calculation (using chord-length parameterization)

class cubic_interpolating_curve(planar_curve_deg3):
    def __init__(self, XY_waypoints: np.array):
        """a minimalistic 2D interpolating curve with chord-length length parametrization
     
        functionalities
        -----------------
        * get_XXX
        * project  ---  calculate the path parameter s* at which a query point is projected to p(s).

        Why not just linear interpolation?
        ---------------------------------
        Many motion control algorithms requires C2 smoothness
        (in order that the second-order derivatives and curvature are meaningfully defined).


        Notes
        -----------------
        * How do I specify the breakpoints?
          No you don't. 
          This data type always uses (metric scaled) chord-length parameterization.
          Experience shows that the resultant curve is typically close to unit-speed.
          Especially when you have reasonably dense samples/ interpolating points
          around high-curvature interval(s).
        
        * By default, 
          the domain of definition starts with 0.0 and ends with the
          approximate total path length
 
        * internally implemented as cubic B-spline with 
          "natural" end condition.

        * The implementation does not modify the XY_waypoint.
          In fact, it will copy the given waypoints.
          (as inherited from the base class)        
        """
        _chord_length_calculator = polyline(XY_waypoints) # which will btw validate XY_waypoints
        self.XY_waypoints = XY_waypoints
        self.idx2arclen = _chord_length_calculator.idx2arclen

        # spl is a shorthand for spline, each is a tuple of 
        # (t --- knots , c --- coefficients, k -- order)
        self.spl_x = splrep(self.idx2arclen, self.XY_waypoints[:,0],s=0)   
        self.spl_y = splrep(self.idx2arclen, self.XY_waypoints[:,1],s=0)
        self.spl_dotx = splder(self.spl_x)
        self.spl_doty = splder(self.spl_y)
        self.spl_ddotx = splder(self.spl_dotx)
        self.spl_ddoty = splder(self.spl_doty)
        # TODO dddotx , etc for trajectory tracking 

    # =================================
    # the required methods
    def get_pos(self, t_eval):
        X_eval = splev(t_eval, self.spl_x)
        Y_eval = splev(t_eval, self.spl_y)
        return np.vstack((X_eval,Y_eval)).T
    def get_tang(self, t_eval):
        dotx = splev(t_eval,self.spl_dotx)
        doty = splev(t_eval,self.spl_doty)
        return np.vstack((dotx,doty)).T

    def get_deri_tang(self, t_eval):
        ddotx = splev(t_eval,self.spl_ddotx)
        ddoty = splev(t_eval,self.spl_ddoty)
        return np.vstack((ddotx,ddoty)).T


class cubic_interpolating_loop(planar_curve_deg3):
    def __init__(self, XY_waypoints: np.array):
      """ periodic cubic interpolating planar curve

      input
      -----------
      XY_waypoints: Nx2 numpy arry

      Notes
      -----------
        * The first waypoint is assumed to be the "wrapping" point.
        * The last waypoint will be automatically joined 
          to the first so please do NOT append the first waypoints 
          at the end `XY_waypoints`.
          This constructor will do it for you.
        * The periodicity can be queried using `.tot_dist`
        * You can also obtain the wrapped curve parameter using `.wrap`
      """
      self.XY_waypoints = np.vstack([
          XY_waypoints,
          XY_waypoints[0].reshape(1,2)
      ])
      _chord_length_calculator = polyline(self.XY_waypoints)
      self.idx2arclen = _chord_length_calculator.idx2arclen # which will btw validate XY_waypoints

      self.spl_xy = CubicSpline(
          self.idx2arclen, 
          self.XY_waypoints, 
          bc_type='periodic',
          #extrapolate=None # the default is sufficient for handling the wrapping
      )
      self.spl_dot = self.spl_xy.derivative(1)
      self.spl_ddot = self.spl_dot.derivative(1)

    # =================================
    # the required methods
    def get_pos(self, t_eval):
        return self.spl_xy(t_eval)
    def get_tang(self, t_eval):
        return self.spl_dot(t_eval)
    def get_deri_tang(self, t_eval):
        return self.spl_ddot(t_eval)

    @classmethod
    def is_periodic(cls):
      return True # just FYI
    
    def wrap(self, s):
      """accept only one test values!
      wrap to [s_min, s_max]

      Make sure you don't confuse `wrap` and `clip` !
      """
      s_wrapped = s
      period = self.tot_dist # calculate s_max-s_min only once
      while s_wrapped < self.s_min:
        s_wrapped += period
      while s_wrapped > self.s_max:
        s_wrapped -= period
      return s_wrapped 


    