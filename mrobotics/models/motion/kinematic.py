import numpy as np
# from .. .. geom.pose2 import pose2

#  everything in SI unit, right-hand notation


class ackermann_car:
    def __init__(self, wheelbase):
        assert isinstance(wheelbase, float) and wheelbase > 1e-2
        self.l = wheelbase
    def ct(self, current_pose: np.array, vx, steering_angle):
        """ continuous-time dynamics  
        state variables: [X_car,Y_car, Yaw_car]

        return
        -----------
        dot_state_vector (1d numpy array, size: 3)        
        """
        dot_X = vx*np.cos(current_pose[2])
        dot_Y = vx*np.sin(current_pose[2])
        dot_theta = vx*np.tan(steering_angle)/self.l
        return np.array([dot_X, dot_Y, dot_theta])

class ackermann_car_actuator(ackermann_car):
    def __init__(self, wheelbase, tau_steering, tau_vx):
        """
        actuation delay in the underlying velocity/ steering servo
        are modelled as 1st order lag each.
        """
        assert tau_steering > 1e-3
        assert tau_vx > 1e-3
        super().__init__(wheelbase)
        self.k_steering = 1/tau_steering
        self.k_vx = 1/tau_vx
    def ct(self, current_state: np.array, vx_cmd, steering_angle_cmd):
        """ continuous-time dynamics  
        state variables: [X_car,Y_car, Yaw_car, steering_angle, vx]

        return
        -----------
        dot_state_vector (1d numpy array, size: 5)
        """
        dot_pose = super().ct(current_state[:3], current_state[4], current_state[3])
        dot_steering = self.k_steering * (steering_angle_cmd - current_state[3])
        dot_vx = self.k_vx * (vx_cmd - current_state[4])
        return np.array([*dot_pose, dot_steering, dot_vx])


