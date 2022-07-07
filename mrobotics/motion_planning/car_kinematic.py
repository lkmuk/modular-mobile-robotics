import numpy as np
from .. piecewise.cubic_onepiece import spline_2d_onepiece
class tg_ackermann_flatness:
    """Trajectory generator for Ackermann kinematic model using differential flatness
    """
    def __init__(self, wheelbase: float, prediction_horizon: float, min_speed: float = 1e-4):
        """represent the flat trajectory by two single-piece cubic polynomials
        
        For technical reasons, there is a lower bound on the longitudinal speed given by
        min_speed
        """
        assert isinstance(wheelbase, float) and wheelbase > 0.0
        assert isinstance(prediction_horizon, float) and prediction_horizon > 0.0
        self.l = wheelbase
        # you can replace the cubic representation with whatever class that provides the same interface
        self.flat_traj_rep = spline_2d_onepiece(0.0, prediction_horizon) 
        self.eps = min_speed**2

    def update_flat_traj(self,p0, v0, pf, vf):
        """calculate the flat output trajectory

        Args:
            p0 (2-numpy array): position vector at t=0
            v0 (2-numpy array): velocity vector at t=0
            pf (2-numpy array): position vector at t=T
            vf (2-numpy array): velocity vector at t=T

            where t:=0 corresponds to the moment when the trajectory starts.
            all vectors corresponds to the same coordinate system.
        
        Calculate the trajectory
        """
        self.flat_traj_rep.fit_pos_vel_ends(p0, v0, pf, vf)

    def compute_traj_x_u(self, t, vel_positive=True):
        """compute the desired state and input trajectories for the given t

        Args:
            t (1d numpy array): the time parameter
            vel_positive (bool, optional): the car-like robot is heading forward?
        """
        # assert self.flat_traj_rep.t0 < t  < self.flat_traj_rep.T-self.flat_traj_rep.t0
        x, y = self.flat_traj_rep.get_pos(t).reshape(-1,2).T
        dotx, doty = self.flat_traj_rep.get_tang(t).reshape(-1,2).T
        ddotx, ddoty = self.flat_traj_rep.get_deri_tang(t).reshape(-1,2).T

        yaw = np.arctan2(doty, dotx)
        if not vel_positive:
            yaw += np.pi

        vel_sq = dotx**2 + doty**2 + self.eps # to avoid singularity in yaw rate and delta
        speed = np.sqrt(vel_sq)
        vx = speed if vel_positive else -speed
        yaw_rate = (ddoty*dotx-ddotx*doty)/vel_sq
        steering_angle = np.arctan(self.l*yaw_rate/vx)

        return np.vstack((x,y,yaw)).T, np.vstack((steering_angle, vx)).T
    def __call__(self):
        """compute u(t=0) 

        return: 
            steering_cmd  (float)
            vx_cmd (float)
        """
        # evaluate U at time t
        _, u0 = self.compute_traj_x_u(0.0)
        return u0[0], u0[1]


