import numpy as np
from .base import planar_curve_deg1
from .search_lhs_bkpt import linear_search as search_left_idx

# import matplotlib.pyplot as plt


def calc_cum_distance(XY_waypts: np.array):
    """calculate polyline cumulative chord length distance at each breakpoint/ waypoint

    Args:
        XY_waypts (np.array): 2 x N_pts

    Returns:
        dist (np.array): 2 x (N_pts-1)
    """
    dist = np.linalg.norm(np.diff(XY_waypts,axis=0), axis=1, ord=2) # sqrt
    return np.cumsum(dist)

class polyline(planar_curve_deg1):
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
    
    def add(self,XY_waypoint):
        # just in case
        XY_in = XY_waypoint.reshape(-1,2)
        
        XY_in_plus_last = np.vstack((self.XY_waypoints[-1], XY_in))
        # update also the distance cache!
        self.idx2arclen = np.append(self.idx2arclen, self.tot_dist+ calc_cum_distance(XY_in_plus_last))
        self.XY_waypoints = np.vstack((self.XY_waypoints, XY_in))
    
    ####################
    #  internal stuffs
    ####################
    def search_first_bkpt_idx(self, arclength):
        idx = search_left_idx(self.idx2arclen, arclength)
        if idx == -1:
            idx = 0 # extrapolate from the first line segment
        elif idx == -2: # force 
            idx = len(self.idx2arclen)-2 # extrapolate from the last line segment
        return idx

    def _get_pos(self, arclength):
        """calculate the xy position along the curve
        Args:
            arclength (float): 
        Returns:
            xy: numpy array (2,)
        """
        idx = self.search_first_bkpt_idx(arclength) 
        p0, p1 = self.XY_waypoints[idx:idx+2]
        scale = (arclength-self.idx2arclen[idx])/(self.idx2arclen[idx+1]-self.idx2arclen[idx])
        return p0 + (p1-p0)*scale

    def _get_utang(self, arclength):
        """unit tangent computation
        Args
        --------
        arclength (float)
        
        Return
        -----------------
        unit_tangent: 1D numpy array (2,)
            
        Remarks
        -----------
        Implementation-defined behavior when evaluated (exactly) at the breakpoints.
        Either the LHS or RHS. 
        (Should be irrelevant in practice due to the nature of floating point)
        """
        idx = self.search_first_bkpt_idx(arclength) 
        p0, p1 = self.XY_waypoints[idx:idx+2]
        p0_to_p1 = p1-p0
        return (p0_to_p1)/np.linalg.norm(p0_to_p1,ord=2)

    def _unify_output_despite_polymorphic_input(self, arclength, method):
        if isinstance(arclength, float):
            return getattr(self, method)(arclength).reshape(1,2)
        else: 
            N = len(arclength)
            xy = np.zeros((N,2))
            for i in range(N):
                xy[i] = getattr(self, method)(arclength[i])
            return xy

    #######################
    #  the required methods
    #######################
    def get_pos(self, arclength):
        return self._unify_output_despite_polymorphic_input(arclength, "_get_pos")
    def get_utang(self, arclength):
        return self._unify_output_despite_polymorphic_input(arclength, "_get_utang")
    def get_tang(self, arclength):
        return self.get_utang(arclength)
    
    #######################
    # override
    #######################
    def _get_pos_utang(self, arclength, shortcut=None):
        """ override the base class for more efficient computation

        Bundled together to save computation
        (specialized for linear interpolation)

        the argument `shortcut` is not used but necessary for ensuring the 
        compatibility with the generic "project" implementation.
        """
        idx = self.search_first_bkpt_idx(arclength) 
        p0, p1 = self.XY_waypoints[idx:idx+2]
        p0_to_p1 = p1-p0
        scale = (arclength-self.idx2arclen[idx])/(self.idx2arclen[idx+1]-self.idx2arclen[idx])
        return p0 + p0_to_p1*scale, (p0_to_p1)/np.linalg.norm(p0_to_p1,ord=2)
    
    # def viz(self, ax=None):
    #     if ax is None:
    #         _, ax = plt.subplots()
    #     ax.plot(*self.XY_waypoints.T, marker='o',ls=':', label='waypoints')
    #     ax.set_aspect('equal')
    #     ax.grid('both')
    #     ax.legend()
            
    def save_as_bin(self, fpath):
        data = np.hstack((self.idx2arclen.reshape(-1,1), self.XY_waypoints)).astype(np.float64)
        data.tofile(fpath)

class polyline_from_bin(polyline):
    def __init__(self, fpath, keep_src_arc_length=False):
        data = np.fromfile(fpath, dtype=np.float64).reshape(3,-1)
        # we will ignore the breakpoints
        data_xy = data[:,1:3]
        super().__init__(data_xy)
        if keep_src_arc_length:
            self.idx2arclen = data[:,0]