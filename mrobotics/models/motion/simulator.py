import numpy as np
from . kinematic import ackermann_car_actuator
import matplotlib.pyplot as plt

class kinematic_car_simulator:
    def __init__(self, step_sz, sim_duration, initial_state, **plant_param_dict):
        """a minimal simulation master containing only the plant physics (simulated using RK4) 
        
        How to use this simulator? see the example.

        the plant of interest has the following states
        1. X-location of the kinematic center of the car (w.r.t. map coordinate system)
        2. Y-location ...
        3. Yaw angle of the car
        4. Steering angle of the front axle (not the steering servo angle)
        5. longitudinal velocity of the vehicle

        The plant inputs are 
        1. desired steering angle
        2. desired longitudinal velocity of the vehicle

        All quantities in SI and right-hand notation

        assume/ limitation of this simulation framework:
        * everything synchronous 
          (i.e. your controller is updated at a fixed time interval, denoted T_ctr
          specifically, T_ctr is a multiple of the solver time step.
        * Relevant if your control updates only every X steps:
          The ZOH state shall be saved externally!
        * Effectively, only explicit integration schemes are supported.
        """
        assert 0 < step_sz < sim_duration, "The total simulation time must be larger than one step size!"
        
        self.step_sz = step_sz
        N_steps = int(np.ceil(sim_duration/step_sz))# excluding time 0
        self.num_steps = N_steps
        self.t_sim_ev = np.linspace(0, step_sz*N_steps, N_steps+1)

        # preallocate the buffer for storing the plant trajectories
        self.X = np.ones((N_steps+1, 5), dtype=float)*np.nan
        self.X[0,:] = initial_state
        self.U = np.ones((N_steps  , 2), dtype=float)*np.nan

        self.plant = ackermann_car_actuator(**plant_param_dict)

        # number of steps completed
        self.step_ptr = 0 

    def get_num_sim_steps(self):
        return self.num_steps

    def get_current_step(self):
        return self.step_ptr
    def get_current_time(self):
        return self.t_sim_ev[self.step_ptr]

    def is_completed(self):
        return self.step_ptr >= self.num_steps

    # ----------------------
    def step_get_state(self):
        """ obtain the current plant state vector
        intended to be used by feedback control.

        No state measurement/ estimation error
        """
        return self.X[self.step_ptr]

    def step_actuate(self, steering_cmd, vx_cmd):
        """ the discrete-time dynamics (using RK4)
        the plant inputs are kept constant during each step.
        must be called at the end of each step!
        """

        state_now = self.X[self.step_ptr]
        k1 = self.plant.ct(state_now, vx_cmd, steering_cmd)
        k2 = self.plant.ct(state_now+(0.5*self.step_sz)*k1, vx_cmd, steering_cmd)
        k3 = self.plant.ct(state_now+(0.5*self.step_sz)*k2, vx_cmd, steering_cmd)
        k4 = self.plant.ct(state_now+(self.step_sz)*k3, vx_cmd, steering_cmd )

        # log the result & increment the step_ptr
        self.U[self.step_ptr] = [steering_cmd, vx_cmd]
        self.step_ptr += 1
        self.X[self.step_ptr] = state_now + (self.step_sz/6)*(k1+ 2*k2+ 2*k3+ k4)

    def viz_trace(self, axs=None):
        """Show all signals as seen by the plant simulator
        axs shall be a list of 5 matplotlib axes (one for each state variables)
        If you don't care about overlay, just keep the default ("None")

        Call "plt.show" yourself only after calling this method!
        """
        if axs is None:
            _, axs = plt.subplots(5,1,sharex=True)
        else:
            assert len(axs)==5
        
        plot_style = {'color': 'k', 'ls':'-', 'label':'simulated'}
        plot_style_U = {'color': 'r', 'ls':':', 'label':'command'}

        ax = axs[0]
        ax.plot(self.t_sim_ev, self.X[:,0], **plot_style)
        ax.set_ylabel('x (m)')

        ax = axs[1]
        ax.plot(self.t_sim_ev, self.X[:,1], **plot_style)
        ax.set_ylabel('y (m)')

        ax = axs[2]
        ax.plot(self.t_sim_ev, self.X[:,2]*180/np.pi, **plot_style)
        ax.set_ylabel('Yaw angle (deg)')

        ax = axs[3]
        ax.plot(self.t_sim_ev, self.X[:,3]*180/np.pi, **plot_style)
        ax.plot(self.t_sim_ev[:-1], self.U[:,0]*180/np.pi, **plot_style_U)
        ax.legend()
        ax.set_ylabel('steering angle (deg)')

        ax = axs[4]
        ax.plot(self.t_sim_ev, self.X[:,4], **plot_style)
        ax.plot(self.t_sim_ev[:-1], self.U[:,1], **plot_style_U)
        ax.legend()
        ax.set_ylabel('$V_x$ (m/s)')

        ax.set_xlabel('t (sec)')

    def viz_pose(self, ax=None):
        """XY-plot of the resultant trajectory
        expect ax to be a matplotlib axis object.

        Call "plt.show" yourself only after calling this method!
        """
        if ax is None:
            _, ax = plt.subplots()
        
        ax.plot(self.X[:,0],self.X[:,1], '-k',label='simulated') 
        ax.set_aspect('equal')
        ax.grid('major')
        ax.legend()
        ax.set_title('Car position trajectory')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

    def save_res_as_mat(self, fpath):
        pass
