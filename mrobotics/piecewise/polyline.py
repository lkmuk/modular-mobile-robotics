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

class waypoints:
    def __init__(self, XY_waypoints):
        """polyline in arc-length paramterization
        expect an numpy array of (N_waypt x 2)
        
        Even after the initialization,
        you can actually add more waypoints, thereby extending the polyline path
        which can be useful for online planning.
                
        assume 2D polyline
        """
        assert XY_waypoints.ndim == 2 and XY_waypoints.shape[1] == 2
        assert XY_waypoints.shape[0] >=2, "if initialized without enough waypoint, code might break!"
        self.XY_waypoints = XY_waypoints
        self.idx2arclen = np.append([0.0],calc_cum_distance(XY_waypoints)) # one-time computation, a sorted array
    

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

    def get_pos(self, arclength,  clip=True):
        # the idx_first_true is meant for avoiding certain computation, if specified, it MUST be the correct one!
        if clip:
            arclength = np.clip(arclength, 0.0, self.get_tot_dist())
        idx = self.search_first_bkpt_idx(arclength) 
        p0, p1 = self.XY_waypoints[idx:idx+2]
        scale = (arclength-self.idx2arclen[idx])/(self.idx2arclen[idx+1]-self.idx2arclen[idx])
        return p0 + (p1-p0)*scale

    def get_utang(self, arclength, clip=True):
        """unit tangent computation

        Args:
            arclength (float)
        Remarks:
        * Works also for 3D case
        """
        # some recomputation if we want to comute get_pos_no_extrapolation and get_utang_no_extrapolation at the same time
        if clip:
            arclength = np.clip(arclength, 0.0, self.get_tot_dist())
        idx = self.search_first_bkpt_idx(arclength) 
        p0, p1 = self.XY_waypoints[idx:idx+2]
        p0_to_p1 = p1-p0
        return (p0_to_p1)/np.linalg.norm(p0_to_p1,ord=2)

    def get_pos_utang(self, arclength, clip=True):
        """ prefer this if you want both!
        """
        if clip:
            arclength = np.clip(arclength, 0.0, self.get_tot_dist())
        idx = self.search_first_bkpt_idx(arclength) 
        p0, p1 = self.XY_waypoints[idx:idx+2]
        p0_to_p1 = p1-p0
        scale = (arclength-self.idx2arclen[idx])/(self.idx2arclen[idx+1]-self.idx2arclen[idx])
        return p0 + p0_to_p1*scale, (p0_to_p1)/np.linalg.norm(p0_to_p1,ord=2)

    def project(self, XY_query, arclength_init_guess, iter_max = 5, soln_tolerance = 0.001):
        """Project a query point to the polyline in an iterative manner

        Args:
            XY_query (2-numpy array)
            arclength_init_guess (float)
            iter_max (int, optional): Defaults to 5.
            soln_tolerance (float, optional): Defaults to 0.001.

        Returns:
            projected_arc_length (float)
            which return nan if failed to converg

        Remarks:
        * allows extrapolation
        * It can also work to some extent for higher order parametric curves.
          ~ linearizing your curve around the current guess
        * Works also for 3D case
        """
        # the update law here is tailored to polyline (formed by joining the waypoints)
        # alway performs extrapolation if needed 
        #   Notes: it's up to the client to make sense of the returned arclength, 
        #   the `is_interior` member function shall be useful in this case
        assert isinstance(iter_max, int) and iter_max > 0
        s_old = -1e16 
        i = 0
        s_new = arclength_init_guess
        converged = False
        while (not converged and i < iter_max):
            # print(f"working on iteration {i}", end=': ')
            s_old = s_new
            curve_pos_now, curve_unit_tang_now = self.get_pos_utang(s_old, clip=False)
            s_new += np.dot(XY_query-curve_pos_now, curve_unit_tang_now) # update law
            # print(f'new iterate {s_new:.3f}')
            converged = np.abs(s_new-s_old) < soln_tolerance
            i+= 1
        if not converged:
            s_new = np.nan # ^.^
        return s_new



    def viz(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(*self.XY_waypoints.T, marker='o',ls=':', label='waypoints')
        ax.set_aspect('equal')
        ax.grid('both')
        ax.legend()
            
