import numpy as np
from .. import wrap_angle
from .. import calc_xy_front_axle, v_front_axle_from_robot_state

def stanley(error_dist: float, error_yaw: float, v_front_axle: float, gain: float, max_steering: float = 1.05):
    """Stanley steering controller

    control objective: stabilize error_dist

    keyword: path following for holonomic systems

    Args:
        error_dist (float): 
            the Frenet y-coordinate of the FRONT AXLE point
            a SIGNED quantity!!!
        error_yaw (float): 
            the yaw error of the robot RELATIVE to the Frenet frame
            (associated with the FRONT AXLE point).
            CCW +ve
        v_front_axle (float): must be +ve !!!
            either the true value or 
            the commanded value.
            
            See the notes for more details on the calculation.

        gain (float): must be +ve
            Feedback gain
        max_steering (float): must be in (0, pi/2), radian
            default = 60 deg

    Returns:
        front_steering_angle (float): radian, CCW +ve

    Notes:
        * The error is computed based on the Frenet frame 
          of the FRONT-axle projection point (not the rear one)!
        * In case you wish to use the measured/ ground-truth value for 
            v_front_axle, you might consider using `v_front_axle_from_robot_state`
        * The path is assumed to have continuous tangent vector (C1).
    """
    front_steering_angle = - np.arctan2(gain*error_dist, v_front_axle) # instead of using atan in order to avoid singularity
    front_steering_angle -= error_yaw
    front_steering_angle = wrap_angle(front_steering_angle)
    front_steering_angle = float(np.clip(front_steering_angle, -max_steering, max_steering))
    return front_steering_angle



class stanley_path_follower:
    def __init__(self, path_to_follow, s_last, wheelbase, steering_max, gain, sampling_period, min_vel_to_enable_steer=0.001):
        """Functor wrapper for the stanley steering controller
        Don't forget!!! 
          The query point, or the path progress pretains to 
          the FRONT axle whereas
          our robot pose (from the simulator) has its reference point chosen 
          to be the rear axle center

        The parameter `min_vel_to_enable_steering` pretains to the front wheel!
        (should be minor though).
        The point is we shall avoid changing steering when the car is almost stationary.
        """
        assert hasattr(path_to_follow, "project") and hasattr(path_to_follow, "get_utang")
        assert isinstance(s_last,float)
        assert isinstance(wheelbase,float) and wheelbase > 0.01
        assert isinstance(sampling_period,float) and sampling_period > 0.0
        assert isinstance(gain,float) and gain > 1e-5
        assert isinstance(steering_max,float) and 1e-3 < steering_max < np.pi/2
        assert isinstance(min_vel_to_enable_steer, float) and 0.0 < min_vel_to_enable_steer < 0.5
        self.path = path_to_follow
        self.s_last = s_last
        self.l = wheelbase
        self.u_max = steering_max
        self.k = gain
        self.f_update = 1/sampling_period # Hz --- for checking plausibility of the progress estimate
        self.v_enable = min_vel_to_enable_steer


    def __call__(self,sim_time, robot_state):
        """ steering command update
        
        robot state vector:
        0. X-position (of the rear axle center point !) w.r.t the map coordinate system
        1. Y-position ...
        2. Yaw of the robot wrt ...
        3. Actual steering angle
        4. Actual forward speed (at the rear axle !)

        return: 
        
        steering_angle_cmd (float)
        s_now, offset_now (float): not essential
        """
        # estimate the Frenet offset
        xy_front = calc_xy_front_axle(robot_state[0:3], self.l)
        s_now, offset_now = self.path.project(XY_query=xy_front, arclength_init_guess=self.s_last, soln_tolerance=0.01)
        # plausibility check: update initial guess (for the last controller update)
        if np.abs(self.s_last-s_now)*self.f_update > 2.0*robot_state[4]:
          # ONLY FOR development
          raise ValueError(f"Abrupt change detected at t = {sim_time:.3f}, rear wheel actual speed: {robot_state[4]:.2f}, previous path progress {self.s_last:.2f} m; estimated to be {s_now:.2f} m,  offset {offset_now:.2f} m.")
        else:
          self.s_last = s_now

        # more required variables
        unit_x_vector_Frenet_wrt_world = self.path.get_utang(s_now, clip=False).reshape(2)
        # angle wrapping deferred to the stanley call
        error_yaw = robot_state[2] - np.arctan2(unit_x_vector_Frenet_wrt_world[1], unit_x_vector_Frenet_wrt_world[0]) 
        vf = v_front_axle_from_robot_state(Vx_actual=robot_state[4], steering_actual=robot_state[3])
        if vf > self.v_enable:
            steering_angle_cmd = stanley(offset_now, error_yaw, gain=self.k, max_steering=self.u_max, v_front_axle=vf)
        else:
            steering_angle_cmd = robot_state[3] # hold
        return float(steering_angle_cmd), s_now, offset_now