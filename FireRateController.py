"""Network controller"""

import numpy as np
from scipy.interpolate import CubicSpline
import scipy.stats as ss
import farms_pylog as pylog


class FiringRateController:
    """zebrafish controller"""

    def __init__(
            self,
            pars
    ):
        super().__init__()

        self.n_iterations = pars.n_iterations
        self.n_neurons = pars.n_neurons
        self.n_muscle_cells = pars.n_muscle_cells
        self.timestep = pars.timestep
        self.times = np.linspace(
            0,
            self.n_iterations *
            self.timestep,
            self.n_iterations)
        self.pars = pars

        self.n_eq = self.n_neurons*4 + self.n_muscle_cells*2 + self.n_neurons * \
            2  # number of equations: number of CPG eq+muscle cells eq+sensors eq
        self.muscle_l = 4*self.n_neurons + 2 * \
            np.arange(0, self.n_muscle_cells)  # muscle cells left indexes
        self.muscle_r = self.muscle_l+1  # muscle cells right indexes
        self.all_muscles = 4*self.n_neurons + \
            np.arange(0, 2*self.n_muscle_cells)  # all muscle cells indexes
        # vector of indexes for the CPG activity variables - modify this
        # according to your implementation
        
        #JULIEN les indices nécessaires pour les equations : les r 0-99 les a 100-199 les s 200-299 les m 300-320
        self.left_v = range(self.n_neurons)
        self.right_v=range(self.n_neurons,2*self.n_neurons)
        self.left_a=range(self.n_neurons*2,3*self.n_neurons)
        self.right_a=range(3*self.n_neurons,4*self.n_neurons)
        self.left_s=range(4*self.n_neurons,5*self.n_neurons)
        self.right_s=range(5*self.n_neurons,6*self.n_neurons)
        self.left_m=range(6*self.n_neurons,6*self.n_neurons+self.n_muscle_cells)
        self.right_m=range(6*self.n_neurons+self.n_muscle_cells,6*self.n_neurons+2*self.n_muscle_cells)
        
        
        pylog.warning(
            "Implement here the vectorization indexed for the equation variables")

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
            self.noise_vec = np.zeros(self.n_neurons*2)

        # zero vector activations to make first and last joints passive
        # pre-computed zero activity for the first 4 joints
        self.zeros8 = np.zeros(8)
        # pre-computed zero activity for the tail joint
        self.zeros2 = np.zeros(2)

    def get_ou_noise_process_dw(self, timestep, x_prev, sigma):
        """
        Implement here the integration of the Ornstein-Uhlenbeck processes
        dx_t = -0.5*x_t*dt+sigma*dW_t
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

        dx_process = np.zeros_like(x_prev)

    def step_euler(self, iteration, time, timestep, pos=None):
        """Euler step"""
        self.state[iteration+1, :] = self.state[iteration, :] + \
            timestep*self.f(time, self.state[iteration], pos=pos)
        return np.concatenate([
            self.zeros8,  # the first 4 passive joints
            self.motor_output(iteration),  # the active joints
            self.zeros2  # the last (tail) passive joint
        ])

    def step_euler_maruyama(self, iteration, time, timestep, pos=None):
        """Euler Maruyama step"""
        self.state[iteration+1, :] = self.state[iteration, :] + \
            timestep*self.f(time, self.state[iteration], pos=pos)
        self.noise_vec = self.get_ou_noise_process_dw(
            timestep, self.noise_vec, self.pars.noise_sigma)
        self.state[iteration+1, self.all_v] += self.noise_vec
        self.state[iteration+1,
                   self.all_muscles] = np.maximum(self.state[iteration+1,
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
        return np.zeros(
            2 *
            self.n_muscle_cells)  # here you have to final active muscle equations for the 10 joints

    
    def ode_rhs(self,  _time, state, pos=None):
        """Network_ODE
        You should implement here the right hand side of the system of equations
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
        #JULIEN matrices Win Wss et Wmc
        w_in=np.zeros(self.n_neurons,2)
        w_ss=np.zeros(self.n_neurons,2)
        w_mc=np.zeros(self.n_muscle_cells,2)
        
        #W_in
        for i in  range(50):
            for j in range(50):
                if i<=j and j-i < 1:
                    w_in[i][j]=1/(j-i+1)
                elif i>j and i-j <=2:
                    w_in[i][j]=1/(i-j+1)
                else:
                    w_in[i][j]=0
        #W_ss
        for i in range(50):
            for i in range(50):
                if i<=j and j-i < 10:
                    w_ss[i][j]=1/(j-i+1)
                elif i>j and i-j <=0:
                    w_ss[i][j]=1/(i-j+1)
                else:
                    w_ss[i][j]=0
        #W_mc
        for i in range(self.n_muscle_cells):
            for j in range(self.n_neurons):
                if j>=i*5 and j<=5*(i+1)-1:
                    w_mc[i][j]=1
                else:
                    w_mc[i][j]=0
        
        

        #dstate pour les fatigue (a)
        self.dstate[self.left_a]=(-state[self.left_a]+self.pars.gamma*state[self.left_v])/self.pars.tau_a #fatigue left
        self.dstate[self.right_a]=(-state[self.right_a]+self.pars.gamma*state[self.right_a])/self.pars.tau_a #fatigue right

        #dsate pour les s (res= resultat de l'interp TO DO!!!!!!)
        res=np.zeros(self.n_neurons,1)
        res=[]
        self.dstate[self.left_s]=(res*(1-state[self.left_s])-state[self.left_s])/self.pars.tau_str
        self.dstate[self.right_s]=(-res*(1-state[self.right_s])-state[self.right_s])/self.pars.tau_str

        #dstate pour les firing rate (r)
        for i in range(self.n_neurons):
            
            tot_in_left=0
            tot_in_right=0
            tot_ss_left=0
            tot_ss_right=0

            for j in range(self.n_neurons):
                tot_in_left+=w_ss[i][j]*state[self.n_neurons+i]
                tot_in_right=w_ss[i][j]*state[i]
                tot_ss_left+=w_ss[i][j]*state[5*self.n_neurons+i]
                tot_ss_right+=w_ss[i][j]*state[4*self.n_neurons+i]
            
            self.dstate[i]=(-state[i]+np.max(0,self.pars.I-self.pars.Idiff-self.pars.b*state[2*self.n_neurons+i]-self.pars.w_inh*tot_in_left-self.pars.w_stretch*tot_ss_left))/self.pars.tau #firing rate gauche
            self.dstate[self.n_neurons+i]=(-state[self.n_neurons+i]+np.max(0,self.pars.I-self.pars.Idiff-self.pars.b*state[3*self.n_neurons+i]-self.pars.w_inh*tot_in_right-self.pars.w_stretch*tot_ss_right))/self.pars.tau #firing rate droite
       

        #dstate pour les muscles
        for i in range(self.n_muscle_cells):
            
            tot_left=0
            tot_right=0
            
            for j in range(self.n_neurons):
                tot_left+=w_mc[i][j]*state[j]
                tot_right=w_mc[i][j]*state[self.n_neurons+j]
            
            self.dstate[6*self.n_neurons+i]=self.pars.act_strength*tot_left*(1-state[6*self.n_neurons+i])/self.pars.tau_a-self[6*self.n_neurons+i]/self.pars.tau_d #muscles gauche
            self.dstate[6*self.n_neurons+self.n_muscle_cells+i]=self.pars.act_strength*tot_right*(1-state[6*self.n_neurons+self.n_muscle_cells+i])/self.pars.tau_a-self[6*self.n_neurons+self.n_muscle_cells+i]/self.pars.tau_d #muscles droit
       
        return self.dstate

