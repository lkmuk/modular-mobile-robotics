import numpy as np

"""
In general, some functions for sampling some points along a path.

Useful for providing constraints when generating trajectories.
"""

class p0v0pfvf_selector:
    def __init__(self, path_obj, initial_progress, update_interval, max_plausible_progress_rate = 30.0):
        """for selecting the terminal position and velocity whenever generating a trajectory

        at the moment, path_obj is of type `polyline.waypoints`
        You can use other trajectory/ path representation as well,
        as long as they have the APIs 
        `get_pos_utang` and `project` implemented.

        """
        # check if the **required** API exists!
        assert isinstance(initial_progress, float)
        assert isinstance(update_interval, float) and update_interval > 0.0
        assert isinstance(max_plausible_progress_rate, float) and max_plausible_progress_rate > 0.0
        self.path = path_obj 
        self.progress = initial_progress
        self.progress_change_ub = max_plausible_progress_rate*update_interval # upper bound

    def update(self, robot_pose, T_horizon, desired_progress_rate):
        """update the progress along the path & regenerate boundaries conditions

        assume
            T_horizon : positive
            desired_progress_rate (m/s): positive , smaller than max_plausible_progress_rate

        return
            p0, v0, pf, vf
        will always extrapolate!

        probably problematic! ---> cutting corners?

        """
        # update the current progress
        arc_length_progress_now = self.path.project(robot_pose[:2], arclength_init_guess=self.progress)[0]
        progress_change = arc_length_progress_now-self.progress
        if np.abs(progress_change) > self.progress_change_ub:
            raise ValueError(f'The progress changes ({progress_change:.2f} m) too abruptly, which is not unreasonable')
        elif progress_change > 0.0:
            self.progress = arc_length_progress_now 
        
        # compute pf, vf
        progress_terminal = self.progress + desired_progress_rate*T_horizon
        xy_f   = self.path.get_pos(progress_terminal)
        utan_f = self.path.get_utang(progress_terminal)
        v_f = desired_progress_rate*utan_f 
        # v_f = desired_progress_rate*utan_f*0.2 # probably shouldn't do this in junction with the single-piece cubic flatness trajectory generator
        
        # compute p0, v0  
        xy_0 = robot_pose[:2]
        v_0 = desired_progress_rate*np.array([np.cos(robot_pose[2]),np.sin(robot_pose[2])])
        # seems more reasonable to use the same initial condition as the current state variables.
        # xy_0   = self.path.get_pos(self.progress)
        # utan_0 = self.path.get_utang(self.progress)
        # v_0 = desired_progress_rate*utan_0


        # return xy_0, v_0, xy_f, v_f
        return map(lambda arr_like_obj: np.array(arr_like_obj).reshape(2), (xy_0, v_0, xy_f, v_f)) # just in case
