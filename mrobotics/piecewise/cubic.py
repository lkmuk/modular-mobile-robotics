import numpy as np
from scipy.interpolate import splrep, splev, splder # splint is not useful in our case
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
