import numpy as np
from scipy.interpolate import splrep, splev, splder # splint is not useful in our case
from mrobotics.piecewise.polyline import polyline # mainly for the API and (heuristic) breakpoint calculation

import matplotlib.pyplot as plt


class interpolating_path2d(polyline):
    def __init__(self, XY_waypoints: np.array):
        """a minimalistic 2D interpolating curve with quasi arc-length parametrization
     
        functionalities
        -----------------
        * get_pos ---   the XY position(s) of the queried path parameter(s)
        * get_utang --- unit tangent vector(s)
        * get_tang ---  the "parametric velocity" 
                        (if the arc length is good enough then the "parametric speed" 
                        should be approximately 1 throughout the domain)
        * get_tang_and_curv --- curvature
        * project  ---  calculate the path parameter s* at which a query point is projected to p(s).

        Why not just linear interpolation?
        ---------------------------------
        Many motion control algorithms requires C2 smoothness
        (in order that the second-order derivatives and curvature are meaningfully defined).


        Notes on applications
        -----------------
        * The curve parameter is only APPROXIMATELY the arc-length.
          If the given waypoints implies a very twisty polyline, 
          this approximation becomes less accurate.

          The user is responsible to ensure 
          * this inaccuracy won't cause any issue in his/her application 
            and/or 
          * ensure successive waypoints are not too jagged. 
          (Maybe helpful: add intermediate waypoints)

        * The domain of definition starts with 0.0 and ends with the
          approximate totoal path length
        * Handling of extrapolation request:
            TODO 

        * internally implemented as cubic B-spline with 
          "not-a-knot" end condition.
        * How do I specify the breakpoints/ knots for 
          the interpolating points?
          No, you don't.
          This implementation uses linear interpolation to estimate
          "appropriate" knots for the interpolating points.

        * The implementation does not modify the XY_waypoint.
          In fact, it will copy the given waypoints.
          (as inherited from the base class)
        """
        # 1. input validation 
        # 2. decide on the knots which is stored in self.idx2arclen
        #    (it should be montonically increasing)
        super().__init__(XY_waypoints)
        self._init_spline_objs()            
    
    # =================================
    # inherited from the base class
    # * is_interior
    # * search_first_bkpt_idx
    # * project
    # * many others ...
    
    # ==================================
    # overriden from the base class
    def get_pos(self, t_eval,  clip=True):
        # TODO clip/ extrapolation handling
        X_eval = splev(t_eval, self.spl_x)
        Y_eval = splev(t_eval, self.spl_y)
        return np.vstack((X_eval,Y_eval)).T
    def get_utang(self,t_eval, clip=True):
        """
        if you believe the arc-length parameterization
        is accurate enough, consider using get_tang instead!
        """
        # TODO clip/ extrapolation handling
        tang_vecs = self.get_tang(t_eval)
        return tang_vecs/np.linalg.norm(tang_vecs, axis=1, ord=2).reshape(-1,1)

    def viz(self, t_eval=None, ax=None, **plt_kwargs):
        if ax is None:
            _, ax = plt.subplots()
        if t_eval is None:
            t_eval = np.arange(0.0,self.get_tot_dist(),0.1) # may skip the last few ones
        XY_eval = self.get_pos(t_eval,clip=False)
        ax.plot(*XY_eval.T, **plt_kwargs)
        ax.plot(*self.XY_waypoints.T, 'o')
        ax.set_aspect('equal')
        ax.grid('both')
        # ax.legend()
    
    def _get_pos_utang(self, t_query, clip=True):
        """ prefer this if you want both!

        t_query: float
            can't be an array!!!

        (provided in order to reuse the `project` method 
        from the base class)
        which also expect only a single query
    
        """
        if clip:
            arclength = np.clip(t_query, 0.0, self.get_tot_dist())
        pos = self.get_pos(t_query).reshape(2)
        utang = self.get_utang(t_query).reshape(2)
        return pos, utang

    # =================================
    # extension
    def get_tang(self,t_eval, clip=True):
        # TODO clip/ extrapolation handling
        dotx = splev(t_eval,self.spl_dotx)
        doty = splev(t_eval,self.spl_doty)
        return np.vstack((dotx,doty)).T

    def get_deri_tang(self, t_eval, clip=True):
        # TODO clip/ extrapolation handling
        ddotx = splev(t_eval,self.spl_ddotx)
        ddoty = splev(t_eval,self.spl_ddoty)
        return np.vstack((ddotx,ddoty)).T

    def get_tang_and_curv(self, t_eval, clip=True):
        """compute tangent and curvature
        use get_tang(t_eval) if you ONLY need the tangent!
        
        inputs
        -------------
        t_eval: 1d np vector of size N

        outputs
        -------------
        tangent_wrt_curve_param: 2 x N
        signed_curvature: 1 x N
            ("left turn" is defined to have +ve curvature)
        """           
        tangent_vecs = self.get_tang(t_eval)
        
        ddot_vecs = self.get_deri_tang(t_eval)
        # unsigned curvature
        #       \| \dot{p} \cross \ddot{p} \|
        # k(t)= ------------------------------
        #        \| \dot{p} \| ^3
        
        # 1. the numerator (specialized for the 2D case, which is already in the correct polarity)
        signed_curvature = tangent_vecs[:,0]*ddot_vecs[:,1] - tangent_vecs[:,1]*ddot_vecs[:,0]
        # 2. the denominator
        signed_curvature /= (np.linalg.norm(tangent_vecs,axis=1, ord=2)**3)
        
        return tangent_vecs, signed_curvature
    
    def get_d_curv_d_arc_length(self, t_eval, clip=True):
        """
        use case: feedback linearization for trajectory tracking of car-like robots

        some conversion by chain rule: dk/ds = (dk/dt)/(ds/dt)
        """
        pass

    def project2(self, XY_query, arclength_init_guess, iter_max = 5, soln_tolerance = 0.001,verbose=False):
        """Project a query point iteratively via "error linearization" (exact Newton).)

        Exactly the same API as `project` (which is based on curve linearization).
        The only difference is the underlying algorithm.

        Since this algorithm requires the curve to be twice-differentiable,
        it wasn't (and cannot be) made available to the `polyline` baseclass.

        It can crash if you have a flat error plateau (wrt the arc-length variable).
        In that case, maybe try project1 instead which is more robust but typically slower.
        """
        raise NotImplementedError()

    def _init_spline_objs(self):
        """
        use cases: 
        * iterative reparameterization
        * adding new waypoints
        """
        # spl is a shorthand for spline, each is a tuple of 
        # (t --- knots , c --- coefficients, k -- order)
        self.spl_x = splrep(self.idx2arclen, self.XY_waypoints[:,0],s=0)   
        self.spl_y = splrep(self.idx2arclen, self.XY_waypoints[:,1],s=0)
        self.spl_dotx = splder(self.spl_x)
        self.spl_doty = splder(self.spl_y)
        self.spl_ddotx = splder(self.spl_dotx)
        self.spl_ddoty = splder(self.spl_doty)
        # dddotx , etc for trajectory tracking see [MS08]
    
    # ======================================
    #  suppressed
    def add(self, XY_waypoint):
        raise NotImplementedError("Forbiddened because it would complicate the implementation. Just construct a new one from scratch.")
