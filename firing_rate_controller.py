"""Network controller"""

import numpy as np
from scipy.interpolate import CubicSpline
import scipy.stats as ss
import farms_pylog as pylog


class FiringRateController:
    """zebrafish controller"""

    def __init__(self, pars):
        super().__init__()

        self.n_iterations = pars.n_iterations
        self.n_neurons = pars.n_neurons
        self.n_muscle_cells = pars.n_muscle_cells
        self.timestep = pars.timestep
        self.times = np.linspace(
            0,
            self.n_iterations * self.timestep,
            self.n_iterations)
        self.pars = pars

        self.n_eq = self.n_neurons * 6 + self.n_muscle_cells * 2  # number of equations: number of CPG eq+muscle cells eq+sensors eq
        self.muscle_l = 6 * self.n_neurons + 2 * np.arange(0, self.n_muscle_cells)  # muscle cells left indexes
        self.muscle_r = self.muscle_l + 1  # muscle cells right indexes
        self.all_muscles = 6 * self.n_neurons + np.arange(0, 2 * self.n_muscle_cells)  # all muscle cells indexes

        # Vector of indexes for the CPG activity variables
        self.left_v = range(self.n_neurons)
        self.right_v = range(self.n_neurons, 2 * self.n_neurons)
        self.left_a = range(2 * self.n_neurons, 3 * self.n_neurons)
        self.right_a = range(3 * self.n_neurons, 4 * self.n_neurons)
        self.left_s = range(4 * self.n_neurons, 5 * self.n_neurons)
        self.right_s = range(5 * self.n_neurons, 6 * self.n_neurons)
        self.left_m = range(6 * self.n_neurons, 6 * self.n_neurons + self.n_muscle_cells)
        self.right_m = range(6 * self.n_neurons + self.n_muscle_cells, 6 * self.n_neurons + 2 * self.n_muscle_cells)

        self.state = np.zeros([self.n_iterations, self.n_eq])  # equation state
        self.dstate = np.zeros([self.n_eq])  # derivative state
        self.state[0] = np.random.rand(self.n_eq)  # set random initial state

        self.poses = np.array([
            0.007000000216066837,
            0.00800000037997961,
            0.008999999612569809,
            0.009999999776482582,
            0.010999999940395355,
            0.012000000104308128,
            0.013000000268220901,
            0.014000000432133675,
            0.014999999664723873,
            0.01600000075995922,
        ])  # active joint distances along the body (pos=0 is the tip of the head)
        self.poses_ext = np.linspace(
            self.poses[0], self.poses[-1], self.n_neurons)  # position of the sensors

        # initialize ode solver
        self.f = self.ode_rhs

        # stepper function selection
        if self.pars.method == "euler":
            self.step = self.step_euler
        elif self.pars.method == "noise":
            self.step = self.step_euler_maruyama
            # vector of noise for the CPG voltage equations (2*n_neurons)
            self.noise_vec = np.zeros(self.n_neurons * 2)

        # zero vector activations to make first and last joints passive
        # pre-computed zero activity for the first 4 joints
        self.zeros8 = np.zeros(8)
        # pre-computed zero activity for the tail joint
        self.zeros2 = np.zeros(2)

    def get_ou_noise_process_dw(self, timestep, x_prev, sigma):
        """
        Implement here the integration of the Ornstein-Uhlenbeck processes
        dx_t = -0.1*x_t*dt+sigma*dW_t
        Parameters
        ----------
        timestep: <float>
            Timestep
        x_prev: <np.array>
            Previous time step OU process
        sigma: <float>
            noise level
        Returns
        -------
        x_t{n+1}: <np.array>
            The solution x_t{n+1} of the Euler Maruyama scheme
            x_new = -0.1*x_prev*dt+sigma*sqrt(dt)*Wiener
        """
        dw = np.sqrt(timestep) * np.random.randn(*x_prev.shape)
        dx_process = -0.1 * x_prev * timestep + sigma * dw
        return x_prev + dx_process

    def step_euler(self, iteration, time, timestep, pos=None):
        """Euler step"""
        self.state[iteration + 1, :] = self.state[iteration, :] + \
                                       timestep * self.f(time, self.state[iteration], pos=pos)
        return np.concatenate([
            self.zeros8,  # the first 4 passive joints
            self.motor_output(iteration),  # the active joints
            self.zeros2  # the last (tail) passive joint
        ])

    def step_euler_maruyama(self, iteration, time, timestep, pos=None):
        """Euler Maruyama step"""
        self.state[iteration + 1, :] = self.state[iteration, :] + \
                                       timestep * self.f(time, self.state[iteration], pos=pos)
        self.noise_vec = self.get_ou_noise_process_dw(
            timestep, self.noise_vec, self.pars.noise_sigma)
        self.state[iteration + 1, list(self.left_v) + list(self.right_v)] += self.noise_vec
        self.state[iteration + 1,
        self.all_muscles] = np.maximum(self.state[iteration + 1,
        self.all_muscles],
                                       0)  # prevent from negative muscle activations
        return np.concatenate([
            self.zeros8,  # the first 4 passive joints
            self.motor_output(iteration),  # the active joints
            self.zeros2  # the last (tail) passive joint
        ])

    def motor_output(self, iteration):
        """
        Here you have to final muscle activations for the 10 active joints.
        It should return an array of 2*n_muscle_cells=20 elements,
        even indexes (0,2,4,...) = left muscle activations
        odd indexes (1,3,5,...) = right muscle activations
        """
        left_muscles = self.state[iteration, self.left_m] * self.pars.act_strength
        right_muscles = self.state[iteration, self.right_m] * self.pars.act_strength
        return np.ravel(np.column_stack((left_muscles, right_muscles)))

    def ode_rhs(self, _time, state, pos=None):
        """Network_ODE without feedback for Exercise 3
        You should implement here the right hand side of the system of equations without feedback terms
        Parameters
        ----------
        _time: <float>
            Time
        state: <np.array>
            ODE states at time _time
        Returns
        -------
        dstate: <np.array>
            Returns derivative of state
        """
        # Connectivity matrices
        W_in = np.zeros((self.n_neurons, self.n_neurons))
        W_mc = np.zeros((self.n_muscle_cells, self.n_neurons))

        # Fill in W_in (CPG to CPG connections)
        for i in range(50):
            for j in range(50):
                if i <= j and j - i <= self.pars.n_asc:
                    W_in[i, j] = 1 / (j - i + 1)
                elif i > j and i - j <= self.pars.n_desc:
                    W_in[i, j] = 1 / (i - j + 1)

        # Fill in W_mc (CPG to muscle cell connections)
        for i in range(self.n_muscle_cells):
            for j in range(self.n_neurons):
                if i * 5 <= j < 5 * (i + 1):
                    W_mc[i, j] = 1

        # Adaptation dynamics
        self.dstate[self.left_a] = (-state[self.left_a] + self.pars.gamma * state[self.left_v]) / self.pars.taua
        self.dstate[self.right_a] = (-state[self.right_a] + self.pars.gamma * state[self.right_v]) / self.pars.taua

        # CPG dynamics
        input_left = self.pars.I - self.pars.b * state[self.left_a]
        input_right = self.pars.I - self.pars.b * state[self.right_a]
        self.dstate[self.left_v] = (-state[self.left_v] + np.sqrt(np.maximum(
            input_left - self.pars.w_inh * np.multiply(W_in.dot(np.ones(state[self.right_v].shape)),
                                                       state[self.right_v]), 0))) / self.pars.tau
        self.dstate[self.right_v] = (-state[self.right_v] + np.sqrt(np.maximum(
            input_right - self.pars.w_inh * np.multiply(W_in.dot(np.ones(state[self.left_v].shape)),
                                                        state[self.left_v]), 0))) / self.pars.tau

        # Muscle dynamics
        self.dstate[self.left_m] = (
                    self.pars.w_V2a2muscle * W_mc.dot(state[self.left_v]) * (1 - state[self.left_m]) / self.pars.taum_a
                    - state[self.left_m] / self.pars.taum_d)
        self.dstate[self.right_m] = (self.pars.w_V2a2muscle * W_mc.dot(state[self.right_v]) * (
                    1 - state[self.right_m]) / self.pars.taum_a
                                     - state[self.right_m] / self.pars.taum_d)

        return self.dstate
