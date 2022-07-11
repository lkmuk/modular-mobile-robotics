import numpy as np
from . import apply_SE2_to_pts


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



# -----------------------------------------------------
#   Useful for holonomic path following control schemes,
#   e.g. Stanley controller
# ----------------------------------------------------
def calc_xy_front_axle(robot_pose, wheelbase):
    """xy-position of the front-axle point in the map coordinate system
    Technically, passing in the entire robot state vector is also fine
    as long as the first three are the pose
    [X_rear_center, Y_rear_center, Yaw, ... whatever ]

    Test 
    calc_xy_front_axle([0,1,0.0    ],1.23)
    calc_xy_front_axle([0,1,np.pi/2],1.23)
    calc_xy_front_axle([0,1,np.pi/3],1.23)
    """
    return apply_SE2_to_pts(
        xy =np.array([[wheelbase,0]]), 
        CCW_angle = robot_pose[2], x_shift=robot_pose[0], y_shift=robot_pose[1]
    ).reshape(2)

def v_front_axle_from_robot_state(Vx_actual, steering_actual):
    """calculate the front axle speed
    
    Vx_actual is the REAR axle's actual speed,
    

    Note on simulation/ why emphasising the inputs are the "actual" quantities?

      The 5-order Ackermann car model makes distinction between the
      "actual" and the "commanded" actuation signals.
      Consequently, the plant/ robot state is augmented as:
      [X_actual, Y_actual, Yaw_actual, steering_actual, Vx_actual]

      While the initial goal was to make the simulation more realistic
      (even though we are still assuming perfect no slip),
      it is also very helpful here:
      it allows us to **avoid algebraic loop**.

      Since my simple simulation framework does not support 
      algebraic loops, if you really insist on using a third-order model 
      (i.e. ignoring actuator dynamics^), you have to implement 
      your own (with the vehicle reference point being the front wheel).
      Such model is not implemented in this toolbox (yet).
      
      ^ Alternatively, consider setting a very small time constants for 
      the actuators.
    """
    return Vx_actual/np.cos(steering_actual)