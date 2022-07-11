import numpy as np
from .search_lhs_bkpt import linear_search as search_left_idx

import matplotlib.pyplot as plt


def calc_cum_distance(XY_waypts: np.array):
    """calculate polyline cumulative path distance at each breakpoint/ waypoint

    Args:
        XY_waypts (np.array): 2 x N_pts

    Returns:
        dist (np.array): 2 x (N_pts-1)
    """
    dist = np.linalg.norm(np.diff(XY_waypts,axis=0), axis=1, ord=2) # sqrt
    return np.cumsum(dist)

class polyline:
    def __init__(self, XY_waypoints):
        """linear interpolator of the waypoints (in exact arc-length paramterization)
        
        inputs
        -----------------
        XY_waypoints: N_waypt x 2 numpy array
            standard convention: 1st column: the X value, 2nd column: Y
        
        Notes on the functionality
        -----------------
        Even after the initialization,
        you can actually add more waypoints, thereby extending the polyline path
        which can be useful for online planning.
                
        extension to 3D? (TODO ?)
        -------------------
        search for all "2"
        """
        assert XY_waypoints.ndim == 2 and XY_waypoints.shape[1] == 2
        assert XY_waypoints.shape[0] >=2, "if initialized without enough waypoint, code might break!"
        self.XY_waypoints = XY_waypoints.copy() # copied to ensure data consistence (cf. idx2arclen)
        self.idx2arclen = np.append([0.0],calc_cum_distance(XY_waypoints)) # one-time computation, a sorted array
    
    def get_num_waypts(self):
        return self.XY_waypoints.shape[0]

    def get_tot_dist(self):
        return float(self.idx2arclen[-1])

    def add(self,XY_waypoint):
        # just in case
        XY_in = XY_waypoint.reshape(-1,2)
        
        XY_in_plus_last = np.vstack((self.XY_waypoints[-1], XY_in))
        # update also the distance cache!
        self.idx2arclen = np.append(self.idx2arclen, self.get_tot_dist()+ calc_cum_distance(XY_in_plus_last))

        self.XY_waypoints = np.vstack((self.XY_waypoints, XY_in))
    

    def is_interior(self, arc_length_test):
        return arc_length_test > 0. and arc_length_test < self.get_tot_dist()

    def search_first_bkpt_idx(self, arclength):
        idx = search_left_idx(self.idx2arclen, arclength)
        if idx == -1:
            idx = 0 # extrapolate from the first line segment
        elif idx == -2: # force 
            idx = len(self.idx2arclen)-2 # extrapolate from the last line segment
        return idx

    def _get_pos(self, arclength,  clip=True):
        """calculate the xy position along the curve

        Args:
            arclength (float): 
            clip (bool, optional): Defaults to True.

        Returns:
            xy: numpy array (2,)
        """
        if clip:
            arclength = np.clip(arclength, 0.0, self.get_tot_dist())
        idx = self.search_first_bkpt_idx(arclength) 
        p0, p1 = self.XY_waypoints[idx:idx+2]
        scale = (arclength-self.idx2arclen[idx])/(self.idx2arclen[idx+1]-self.idx2arclen[idx])
        return p0 + (p1-p0)*scale

    def _get_utang(self, arclength, clip=True):
        """unit tangent computation

        Args:
            arclength (float)
            clip (bool, optional): Defaults to True.

        Return:
            unit_tangent: numpy array (2,)
            
        Remarks:
        * Implementation-defined behavior when evaluated (exactly) at the breakpoints.
          Either the LHS or RHS. 
          (Should be irrelevant in practice due to the nature of floating point)
        * Works also for 3D case
        * actually, clipping or not is irrelevant for linear interpolation
        * We delibrately do not implement a get_tang method because
          for linear interpolation it's always going to be a unit tangent.
          (If it is well-defined)
        """
        # some recomputation if we want to comute get_pos_no_extrapolation and get_utang_no_extrapolation at the same time
        if clip:
            arclength = np.clip(arclength, 0.0, self.get_tot_dist())
        idx = self.search_first_bkpt_idx(arclength) 
        p0, p1 = self.XY_waypoints[idx:idx+2]
        p0_to_p1 = p1-p0
        return (p0_to_p1)/np.linalg.norm(p0_to_p1,ord=2)

    def _unify_output_despite_polymorphic_input(self, arclength, method, clip):
        if isinstance(arclength, float):
            return getattr(self, method)(arclength, clip=clip).reshape(1,2)
        else: 
            N = len(arclength)
            xy = np.zeros((N,2))
            for i in range(N):
                xy[i] = getattr(self, method)(arclength[i], clip=clip)
            return xy

    def get_pos(self, arclength, clip=True):
        """calculate the xy position along the curve

        Args:
            arclength (float or 1d numpy array of length N): 
            clip (bool, optional): Defaults to True.

        Returns:
            xy: numpy array (shape: N, 2)
                This also holds if you pass in a float 
                as `arclength` (with N=1).
        """
        return self._unify_output_despite_polymorphic_input(arclength, "_get_pos", clip)

    def get_utang(self, arclength, clip=True):
        """calculate the tangent vector(s) along the curve

        Args:
            arclength (float or 1d numpy array of length N): 
            clip (bool, optional): Defaults to True.

        Returns:
            xy: numpy array (shape: N, 2)
                This also holds if you pass in a float 
                as `arclength` (with N=1).
        """
        return self._unify_output_despite_polymorphic_input(arclength, "_get_utang", clip)
    

    def _get_pos_utang(self, arclength, clip=True):
        """ prefer this if you want both!

        Bundled together to save computation
        (specialized for linear interpolation)
        """
        if clip:
            arclength = np.clip(arclength, 0.0, self.get_tot_dist())
        idx = self.search_first_bkpt_idx(arclength) 
        p0, p1 = self.XY_waypoints[idx:idx+2]
        p0_to_p1 = p1-p0
        scale = (arclength-self.idx2arclen[idx])/(self.idx2arclen[idx+1]-self.idx2arclen[idx])
        return p0 + p0_to_p1*scale, (p0_to_p1)/np.linalg.norm(p0_to_p1,ord=2)

    def project(self, XY_query, arclength_init_guess, iter_max = 5, soln_tolerance = 0.001, verbose=False):
        """Project a query point iteratively via "curve linearization".

        Args:
            XY_query (2-numpy array)
            arclength_init_guess (float)
            iter_max (int, optional): Defaults to 5.
            soln_tolerance (float, optional): Defaults to 0.001.
            verbose (bool): print out the intermediate results ?

        Returns:
            projected_arc_length (float)
                "s" will be returned as nan if failed to converge
            
            projected_distance (float)
                calculated based on the last iterate.
                Notice that it is a signed quantity ---
                defined +ve in the y-axis of the Frenet frame.
            
        Remarks:
        * Even if it appears to converge,
          users are responsible to make sense of the returned values.
          (Read on to see why) 
          
        * abrupt changes relative to the initial guess
            are likely caused by
            
            1. bad initial estimate,
            2. multiple (neighboring) local minimizers and/or

            3. The true projected distance is too far
                (so large that are comparable to the instantaneous turning radius)
                See [Paden16]

                EDIT: This CAN be relevant but NOT in our "curve linearization" case.

        * This algorithm will perform extrapolation if needed.
          In case, this is unacceptable, consider using the 
          `is_interior` member function to verify the result.

        * The "curve linearization" algorithm can also work 
          for higher-order curves
          (provided that they are unit-speed or at least quasi unit-speed curves).
          However, you might consider using `project2` instead 
          (which is based on Newton's method) because
          `project2` typically converges much faster if you 
          have a good initial guess near a distinct local minimum.
        """
        assert isinstance(iter_max, int) and iter_max > 0
        s_old = -1e16 
        i = 0
        s_new = arclength_init_guess
        converged = False
        while (not converged and i < iter_max):
            if verbose:
                print(f"working on iteration {i}", end=': ')
            s_old = s_new
            curve_pos_now, curve_unit_tang_now = self._get_pos_utang(s_old, clip=False)
            delta_XY = XY_query-curve_pos_now
            s_increment = np.dot(delta_XY, curve_unit_tang_now)
            s_new += s_increment # update law
            if verbose:
                print(f'new iterate {s_new:.3f}')
            converged = np.abs(s_new-s_old) < soln_tolerance
            i+= 1

        # calculate the projected distance
        final_unit_normal_vec = np.array([-curve_unit_tang_now[1], curve_unit_tang_now[0]]) # rotation CCW by 90 deg
        projected_distance =  np.dot(delta_XY, final_unit_normal_vec)
        
        if not converged:
            s_new = np.nan # ^.^
        return s_new, projected_distance



    def viz(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(*self.XY_waypoints.T, marker='o',ls=':', label='waypoints')
        ax.set_aspect('equal')
        ax.grid('both')
        ax.legend()
            
