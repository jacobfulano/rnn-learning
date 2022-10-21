import numpy as np
import matplotlib.pyplot as plt

# analysis
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import LinearRegression
from scipy import stats, interpolate
from scipy import linalg as LA

# miscellaneous
from tqdm import tqdm
from itertools import cycle
from copy import deepcopy
import logging
import warnings
import dataclasses
from dataclasses import dataclass
from typing import Optional, List

# custom
from algorithms.base import LearningAlgorithm
from rnn import RNNparams, RNN
from task import Task
from utils.functions import f, df, theta, rgetattr

            
class REINFORCE(LearningAlgorithm):
    
    """
    REINFORCE 
    
    This is a reinforcement learning (RL) rule based on the paper "Biologically plausible learning in 
    recurrent neural networks reproduces neural dynamics observed during cognitive tasks" (Miconi 2017)
    and "Local online learning in recurrent networks with random feedback" (Murray 2019)
    
    Args:
        rnn (RNN): RNN object
        sig_xi (float): noise scale used in learning rule
        tau_reward (float): timescale of reward memory
        apply_to (list): list of weights to apply learning rule, e.g. 'w_rec' or 'w_in'
        online (bool): whether learning rule is online (update every step) or offline (update at end of trial)
            
    Variables used to keep track of derivatives:
        p: eligibility trace for the recurrent weights
            
    TODO: Currently only implemented for w_rec. Need to implement for w_in, w_fb and w_out. Also could implement sporadic reward    
    """
    
    def __init__(self, rnn: RNN, tau_reward: float, apply_to: List[str]=['w_rec'], online: bool = True, error_fn: str = 'distance') -> None:
        
        
        
        # Initialize learning variables
        self.rnn = rnn
        self.p = np.zeros((self.rnn.n_rec, self.rnn.n_rec))
        self.p_fb = np.zeros((self.rnn.n_rec, self.rnn.n_out))
        
        #self.dw_in = np.zeros((self.rnn.n_rec, self.rnn.n_in)) # TO DO
        self.dw_rec = np.zeros((self.rnn.n_rec, self.rnn.n_rec))
        #self.dw_out = np.zeros((self.rnn.n_out, self.rnn.n_rec)) # TO DO
        self.dw_fb = np.zeros((self.rnn.n_rec, self.rnn.n_out))
        
        # variables necessary for this RL algorithm
        self.tau_reward = tau_reward # timescale of reward
        self.r_av = [] # should be a vector dependent on the task
        self.r_av_prev = [] # should be a vector dependent on the task
        self.rnn.r_current = 0
        self.all_tasks = [] # keeps track of unique tasks (when training with multiple tasks)
        self.task_idx = None
        
        assert apply_to != [], 'Must specify which weights to apply learning rule to with "apply_to"'
        
        # check that weight flags match weights in rnn
        for a in apply_to:
            assert a in ['w_in','w_rec','w_out','w_fb'], "specified weights must be selected from ['w_in','w_rec','w_out','w_fb']"
            
        # TO DO: incorporate for w_in, w_out, w_fb
        if 'w_in' in apply_to:
            raise Exception("REINFORCE does not currently work for w_in")
        #    assert rnn.eta_in, "eta_in must be specified if learning is occurring in w_in"
        if 'w_rec' in apply_to:
            assert rnn.eta_rec, "eta_rec must be specified if learning is occurring in w_rec"
        if 'w_out' in apply_to:
            raise Exception("REINFORCE does not currently work for w_out")
        #    assert rnn.eta_out, "eta_out must be specified if learning is occurring in w_out"
        if 'w_fb' in apply_to:
            assert rnn.eta_fb, "eta_fb must be specified if learning is occurring in w_fb"
        
        self.name='REINFORCE'
        self.apply_to = apply_to
        self.online = online
        self.error_fn = error_fn
                
        #assert apply_to[0] == 'w_rec', 'REINFORCE only currently implemented for w_rec' 
        
        # TO DO:
        # * bonus
        # * learning rule for w_out
        # * learning rule for w_fb
        # * learning rule for w_in
        # * sporadic reward at end of trial as alternate learning rule
        
        # self.bonus = 0
        
    
    def update_learning_vars(self, index: int, task: Task):
        
        """
        Update variables associated with learning
        
        Args:
            index (int): trial step
            task (Task): task object that specifies targets, trial duration, etc.
        
        Variables used to keep track of derivatives
            dw_rec: change in the recurrent weights
            
        TODO: Implement for
            dw_out: change in the output weights
            dw_in: change in input weights
        """
        
        """ Keep separate running average for each task """
        if index == 0: # this only needs to be calculated at the beginning of the trial
            self.task_idx = self._track_tasks(task)
        
        task_idx = self.task_idx
                    
        
        # pointer for convenience
        rnn = self.rnn
        t_max = task.trial_duration

            
        
        
        """ update must include noise rnn.xi inject to network recurrent layer """
        if 'w_rec' in self.apply_to: 
            self.p = (1-rnn.dt/rnn.tau_rec)*self.p 
            self.p += np.outer(rnn.xi*rnn.df(rnn.u), rnn.h_prev)*rnn.dt/rnn.tau_rec
        
        if 'w_fb' in self.apply_to:
            self.p_fb = (1-rnn.dt/rnn.tau_rec)*rnn.dt*self.p_fb
            self.p_fb += np.outer(rnn.xi*rnn.df(rnn.u), rnn.pos)*rnn.dt/rnn.tau_rec
            

        # BONUS
#         if index > task.trial_duration-np.round(task.trial_duration/4): # end of trial
#             #norm_av += np.linalg.norm(pos[tt+1]-y_)**2
#             if  np.linalg.norm(rnn.pos-task.y_target)**2 < 0.5: #0.35: # doesn't happen often?
#                 #bonus = bonus_amount
#                 self.bonus += 1
        
        """ Reward as scalar error """
        rnn.err = self._error(task,index)
        rnn.r_current = -(np.linalg.norm(rnn.err))**2 # + self.bonus  
       
        rnn.reward = np.copy(rnn.r_current)  # for plotting purposes, keep track of reward
                    
        if self.online:
            
            self.dw_rec = rnn.eta_rec * (rnn.r_current - self.r_av[task_idx])*self.p/t_max # should this be inside if statement?
            
            if 'w_rec' in self.apply_to: 
                rnn.w_rec = rnn.w_rec + self.dw_rec
                
            if 'w_fb' in self.apply_to:
                self.dw_fb = rnn.eta_fb * (rnn.r_current - self.r_av[task_idx]) * self.p_fb/t_max
                rnn.w_fb = rnn.w_fb + self.dw_fb
                
        if not self.online:
            
            """ running sum of update """
            if 'w_rec' in self.apply_to: 
                self.dw_rec += rnn.eta_rec * (rnn.r_current - self.r_av[task_idx]) * self.p/t_max
                
            if 'w_fb' in self.apply_to:
                self.dw_fb += rnn.eta_fb * (rnn.r_current - self.r_av[task_idx]) * self.p_fb/t_max
                
                
        # TO DO: ADD FOR w_in, w_out and w_fb
                
        """ At the end of the trial """
        if not self.online and index == task.trial_duration-1:
            
            """ could also have sporadic reward at the end of the trial """
            #self.dw_rec = rnn.eta_rec * (rnn.r_current - self.r_av[task_idx])*self.p
    
            if 'w_rec' in self.apply_to: 
                rnn.w_rec = rnn.w_rec + self.dw_rec
                
            if 'w_fb' in self.apply_to:
                rnn.w_fb = rnn.w_fb + self.dw_fb
        
        """ At end of trial, update average reward"""
        if index == task.trial_duration-1:
            
            self.r_av_prev[task_idx] = self.r_av[task_idx]
            self.r_av[task_idx] = self.r_av_prev[task_idx] + (rnn.dt/self.tau_reward) * (rnn.r_current-self.r_av_prev[task_idx])
            
            self.reset_learning_vars() # important for offline learning
            
            

    def reset_learning_vars(self):
        
        """ Reset variables """
        self.p = np.zeros((self.rnn.n_rec, self.rnn.n_rec))
        self.p_fb = np.zeros((self.rnn.n_rec, self.rnn.n_out))
        
        self.dw_rec = np.zeros((self.rnn.n_rec, self.rnn.n_rec))
        self.dw_fb = np.zeros((self.rnn.n_rec, self.rnn.n_out))

        self.rnn.r_av = []
        self.rnn.r_av_prev = []
        self.rnn.r_current = 0
     
    
    def _track_tasks(self,task):
            
        """ Keep track of which task is being used for training
        Note that this is specific for REINFORCE algorithm
        """
        
        task_bool = [np.array_equal(ts.y_target,task.y_target) for ts in self.all_tasks]

        if np.any(task_bool):
            task_idx = np.argwhere(task_bool).squeeze()

        else:
            self.all_tasks.append(task)
            self.r_av.append(0)
            self.r_av_prev.append(0)
            
            task_idx = len(self.all_tasks)-1
            
        return task_idx
    
    
    def _error(self,task,index):
        
        """
        Define error for a given task
        
        1. Distance to target
            This is the vector to the target
            
        2. Scaled distance to target
            This is the vector to the target scaled by the remaining number of timesteps in the trial
            
        3. Velocity
            This treats the error as the difference between the correct velocity vector (distance to the target / time) 
            and the generated velocity vector
            
        TODO: allow for error to be defined by target vector (e.g. a straight line)
        """
        
        if self.error_fn=='distance':
            error = task.y_target - self.rnn.pos
        
        if self.error_fn=='scaled_distance':
            error = (1/(task.trial_duration-index)) * (task.y_target - self.rnn.pos)
        
        if self.error_fn=='velocity':
        
            assert self.rnn.velocity_transform, 'velocity_transform must equal True'
            error = (task.y_target - self.rnn.pos) - self.rnn.vel
        
        return error
    
    
    
    
    
    def print_params(self) -> None:
        
        """ Print Hyperparameters """
        for k in ['apply_to', 'online']:
                print(k,': ',vars(self)[k])
    