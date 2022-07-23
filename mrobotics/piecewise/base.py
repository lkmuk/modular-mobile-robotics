from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class planar_curve_deg1(ABC):
    """ deg1 means that the curve is AT LEAST (typically piecewise) continuous.
    in general, we are interested in interpolating curves.

    required attributes
    -----------------------------
    * idx2arclen: 1D numpy array, size = number of waypoints, must be strictly increasing!
    * XY_waypoints
    """
    @classmethod
    def is_extendable_from_tail(cls):
        return hasattr(cls, "add")

    ########################
    #  stuff related to the domain of definition
    ########################
    def is_periodic(cls) -> bool:
        return False # default value 
    @property
    def s_min(self) -> float:
        return self.idx2arclen[0]
    @property
    def s_max(self) -> float:
        return self.idx2arclen[-1]
    @property
    def tot_dist(self) -> float:
        """
        in the curve domain, not necessarily in the arclength parameterization!

        if `self.is_periodic`, then this shall be interpreted as the periodicity.
        """
        return self.s_max - self.s_min
    def is_interior(self, s_test):
        """ test if the test curve parameter is within the curve domain

        please give only 1 test value!
        """
        if self.is_periodic():
            return True
        else:
            return self.s_min < s_test < self.s_max
    def clip(self, s_test):
        """
        okay to pass in multiple test value at once

        Potential uses: 
        if you want to forbid extrapolation, always call
        mycurve_obj.get_XXX(mycurve_obj.clip(s_test))
        """
        return np.clip(s_test, self.s_min, self.s_max)
    def get_num_waypts(self):
        return len(self.idx2arclen)
    def get_num_pieces(self):
        return self.get_num_waypts - 1


    #########################################
    # Evaluating the curve / its derivatives
    #
    #  API 
    #  1. You can either pass in a single test value
    #     or an 1D array of test values
    #  2. In any case, the output
    #     is a 2D array, with each row for each test value
    #     (in the same order as your input)
    #  3. Extrapolation is always allowed.
    #     See also `clip`, if you want otherwise.
    #########################################
    @abstractmethod
    def get_pos(self, s_test) -> np.array:
        """calculate the position of the curve point(s)
        return
        ----------
        shape Nx2
        
        notes
        -------------
        extrapolation allowed
        """
        pass
    @abstractmethod
    def get_tang(self,s_test) -> np.array:
        """ calculate the tangent(s) (not necessarily normalized)
        return
        ----------
        shape Nx2
        
        notes
        -------------
        extrapolation allowed
        """
        pass
    def get_utang(self, s_test, shortcut=False) -> np.array:
        """calculate unit tangents
        return
        ----------
        shape Nx2
        
        notes
        -------------
        extrapolation allowed

        In most foreseen subclasses,
        we will use chord-length parameterization,
        which typically gives close to unit-speed curve.
        So if normalization is NOT critical, we can just bypass normalization
        """
        tangents = self.get_tang(s_test)
        if shortcut:
            return tangents
        else:
            speeds = np.linalg.norm(tangents, ord=2, axis=1, keepdims=True)
            return tangents/speeds
    
    def _get_pos_utang(self, s_test, shortcut=True) -> (np.array, np.array):
        """
        input
        ----------
        t_query: float
            can't be an array!!!

        notes
        ----------
        conceived mainly for the `project` method below
        Non-standard API! (so it's restricted to internal use, note the `_` prefix)

        override this e.g. if your subclass
        needs to first normalize the curve parameters.
        (so the curve normalization is only performed once)

        """
        return self.get_pos(s_test).reshape(2), self.get_utang(s_test, shortcut=shortcut).reshape(2)

    ####################################
    # more useful stuff
    ###################################
    def project(self, XY_query, arclength_init_guess, iter_max = 5, soln_tolerance = 0.001, verbose=False, skip_normalization=True):
        """Project a query point into the Frenet representation by via successive curve linearization.

        Args
        ----------------
        XY_query (2-numpy array)
        arclength_init_guess (float)
        iter_max (int, optional): Defaults to 5.
        soln_tolerance (float, optional): Defaults to 0.001.
        verbose (bool): print out the intermediate results ?

        Returns
        -----------------
        projected_arc_length (float)
            
        projected_distance (float)
            calculated based on the last iterate.
            Notice that it is a signed quantity ---
            defined +ve in the y-axis of the Frenet frame.

            aka crosstrack error
            
        Remarks/ pitfalls
        ------------------
        The "successive curve linearization" method 
        has geometric intuition as suggested by its name.
        The curve parameter is EXPECTED to be the arc-length
        or at least the curve is close to unit length
        (otherwise the convergency is expected to be slower)

        The users are always responsible for 
        validating the returned values because 
        * it might not converged yet (hint: check the crosstrack error)
        * Extrapolation is always allowed (and you might not want it).
        * There might be multiple "practically" sound solutions,
          especially when 

          * the (estimated) crosstrack error is large AND
          * the curve has relatively high curvature around the query point.

          Mathematically, this can be seen as discontiuity in the query space,
          see [Paden16, Fig. 5].
          
          In these cases, it might be impractical to identify all potential candidates.
          So in any case, you should ensure the initial guess that you give is reasonably close.

        References
        --------------
        Paden16
            Paden, B., Čáp, M., Yong, S. Z., Yershov, D., & Frazzoli, E. (2016). 
            A survey of motion planning and control techniques for self-driving urban vehicles. 
            IEEE Transactions on intelligent vehicles, 1(1), 33-55.
        """
        assert isinstance(iter_max, int) and iter_max > 0
        s_old = -1e16 
        i = 0
        s_new = arclength_init_guess
        converged = False
        while (not converged and i < iter_max):
            s_old = s_new
            curve_pos_now, curve_unit_tang_now = self._get_pos_utang(s_old, shortcut=skip_normalization)
            delta_XY = XY_query-curve_pos_now
            s_increment = np.dot(delta_XY, curve_unit_tang_now)
            s_new += s_increment # update law (assuming exact/ the curve being close to unit-speed)
            if verbose:
                print(f'finished iteration {i} with new iterate {s_new:.3f}')
            converged = np.abs(s_new-s_old) < soln_tolerance
            i+= 1

        # calculate the projected distance
        final_unit_normal_vec = np.array([-curve_unit_tang_now[1], curve_unit_tang_now[0]]) # rotation CCW by 90 deg
        projected_distance =  np.dot(delta_XY, final_unit_normal_vec)
        
        # if not converged:
        #     s_new = np.nan # ^.^  ---- worse then not doing anything
        return s_new, projected_distance

    def viz(self, s_eval=None, ax=None, **plt_kwargs):
        """
        invoke plt.show() or plt.savefig(...) yourself
        """
        if ax is None:
            _, ax = plt.subplots()
        if s_eval is None:
            s_eval = np.arange(self.s_min, self.s_max, 0.1) # may skip the last few ones
        XY_eval = self.get_pos(s_eval)
        ax.plot(*XY_eval.T, **plt_kwargs)
        ax.plot(*self.XY_waypoints.T, 'o',label="waypoints")
        ax.set_aspect('equal')
        ax.grid('both')
        # ax.legend()

class planar_curve_deg2(planar_curve_deg1):
    """
    deg2 means that the curve has AT LEAST (typically piecewise) continuous tangent vector field.
    """
    @abstractmethod
    def get_deri_tang(self, s_test) -> np.array:
        pass

    def get_tang_and_curv(self, s_test) -> (np.array, np.array):
        """compute tangent and curvature

        use get_tang(s_test) if you ONLY need the tangent!
        
        inputs
        -------------
        s_test: 1d np vector of size N

        outputs
        -------------
        tangent_wrt_curve_param: 2 x N
        signed_curvature: 1 x N
            ("left turn" is defined to have +ve curvature)
        """           
        tangent_vecs = self.get_tang(s_test)
        
        ddot_vecs = self.get_deri_tang(s_test)
        # unsigned curvature
        #       \| \dot{p} \cross \ddot{p} \|
        # k(t)= ------------------------------
        #        \| \dot{p} \| ^3
        
        # 1. the numerator (specialized for the 2D case, which is already in the correct polarity)
        signed_curvature = tangent_vecs[:,0]*ddot_vecs[:,1] - tangent_vecs[:,1]*ddot_vecs[:,0]
        # 2. the denominator
        signed_curvature /= (np.linalg.norm(tangent_vecs,axis=1, ord=2)**3)
        
        return tangent_vecs, signed_curvature
    
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

class planar_curve_deg3(planar_curve_deg2):
    """
    deg3 means that the curve is has a polynomial degree of 3 or even more
    """
    def get_d_curv_d_arc_length(self, s_eval, clip=True):
        """
        use case: feedback linearization for trajectory tracking of car-like robots

        some conversion by chain rule: dk/ds = (dk/dt)/(ds/dt)
        """
        pass
