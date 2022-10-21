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


class RFLO(LearningAlgorithm):
    
    """
    Random Feedback Local Online Learning
    
    RFLO is a supervised learning rule for RNNs that is an approximation to RTRL (Real Time Recurrent Learning) and is
    related to BPTT (Backpropagation Through Time).
    
    This implementation is adapted from Murray 2019 "Local online learning in recurrent networks with random feedback"
    (see https://github.com/murray-lab/rflo-learning for more details)
    """
    
    def __init__(self, rnn: RNN, apply_to: List[str]=['w_rec'], online: bool = True, weight_transport: bool = True, error_fn: str = 'scaled_distance') -> None:
        
        """
        Initialize learning rule and set which weights to apply to
        
        Args:
            rnn (RNN): RNN object
            apply_to (list): list of weights to apply learning rule, e.g. 'w_rec' or 'w_in'
            online (bool): whether learning rule is online (update every step) or offline (update at end of trial)
            weight_transpot (bool): if True, updare `w_m` every time `w_out` is updated
            
        Variables used to keep track of derivatives:
            p: eligibility trace for the recurrent weights
            q: eligibility trace for the input weights
            p_fb: eligibility trace for the feedback weights
        """
        
        # Initialize learning variables
        self.rnn = rnn
        self.p = np.zeros((self.rnn.n_rec, self.rnn.n_rec))
        self.q = np.zeros((self.rnn.n_rec, self.rnn.n_in))
        self.p_fb = np.zeros((self.rnn.n_rec, self.rnn.n_out))
        
        self.dw_in = np.zeros((self.rnn.n_rec, self.rnn.n_in))
        self.dw_rec = np.zeros((self.rnn.n_rec, self.rnn.n_rec))
        self.dw_out = np.zeros((self.rnn.n_out, self.rnn.n_rec))
        self.dw_fb = np.zeros((self.rnn.n_rec, self.rnn.n_out))
        
        
        assert apply_to != [], 'Must specify which weights to apply learning rule to with "apply_to"'
        
        # check that weight flags match weights in rnn
        for a in apply_to:
            assert a in ['w_in','w_rec','w_out','w_fb'], "specified weights must be selected from ['w_in','w_rec','w_out','w_fb']"
            
        if 'w_in' in apply_to:
            assert rnn.eta_in, "eta_in must be specified if learning is occurring in w_in"
        if 'w_rec' in apply_to:
            assert rnn.eta_rec, "eta_rec must be specified if learning is occurring in w_rec"
        if 'w_out' in apply_to:
            assert rnn.eta_out, "eta_out must be specified if learning is occurring in w_out"
        if 'w_fb' in apply_to:
            assert rnn.eta_fb, "eta_fb must be specified if learning is occurring in w_fb"
        
        
        self.name='RFLO'
        self.apply_to = apply_to
        self.online = online
        self.weight_transport = weight_transport
        self.error_fn = error_fn
        
                
    
    def update_learning_vars(self, index: int, task: Task):
        
        """
        Update variables associated with learning
        
        Args:
            index (int): trial step
            task (Task): task object that specifies targets, trial duration, etc.
        
        Variables use to keep track of derivatives
            dw_out: change in the output weights
            dw_rec: change in the recurrent weights
            dw_in: change in input weights
            dw_fb: change in feedback weights
            
        TODO: I need to be able to update w_m (e.g. setting equivalent to w_out if w_out is learned) 
        """
        
        # pointer for convenience
        rnn = self.rnn
        t_max = task.trial_duration
        
        
        """ Error based on final target position """
        rnn.err = self._error(task,index)
        
        rnn.loss = 0.5*np.linalg.norm(rnn.err)**2
        
        
        if 'w_rec' in self.apply_to: 
            self.p = (1-rnn.dt/rnn.tau_rec)*self.p
            self.p += np.outer(rnn.df(rnn.u), rnn.h_prev)*rnn.dt/rnn.tau_rec
        if 'w_in' in self.apply_to:
            self.q = (1-rnn.dt/rnn.tau_rec)*self.q
            self.q += np.outer(rnn.df(rnn.u), rnn.x_in_prev)*rnn.dt/rnn.tau_rec  
        if 'w_fb' in self.apply_to:
            self.p_fb = (1-rnn.dt/rnn.tau_rec)*self.p_fb
            self.p_fb += np.outer(rnn.df(rnn.u), rnn.pos)*rnn.dt/rnn.tau_rec
            
            # TODO: Correct for velocity transformation (check)
            #self.p_fb += np.outer(rnn.df(rnn.u), rnn.y_prev)/rnn.tau_rec


        """ Online Update """
        if self.online:
            
            # TODO: Check factor in derivative due to velocity transform
            #if rnn.velocity_transform:
            #    dw_out *= rnn.dt_vel/rnn.tau_vel
            #    dw_rec *= rnn.dt_vel/rnn.tau_vel
            #    dw_in *= rnn.dt_vel/rnn.tau_vel
            
            
                    
            if 'w_rec' in self.apply_to: 
                self.dw_rec = rnn.eta_rec * np.outer(np.dot(rnn.w_m, rnn.err),np.ones(rnn.n_rec)) * self.p/t_max
                rnn.w_rec = rnn.w_rec + self.dw_rec
            if 'w_in' in self.apply_to:
                self.dw_in = rnn.eta_in * np.outer(np.dot(rnn.w_m, rnn.err),np.ones(rnn.n_in)) * self.q/t_max
                rnn.w_in = rnn.w_in + self.dw_in
            if 'w_fb' in self.apply_to:
                self.dw_fb = rnn.eta_fb * np.outer(np.dot(rnn.w_m, rnn.err),np.ones(rnn.n_out)) * self.p_fb/t_max
                rnn.w_fb = rnn.w_fb + self.dw_fb
                
            # Note that w_out is updated at the end, so that w_m is updated _after_ all the other weights
            if 'w_out' in self.apply_to:
                self.dw_out = rnn.eta_out*np.outer(rnn.err, rnn.h)/t_max
                rnn.w_out = rnn.w_out + self.dw_out
                
                # update w_m as well
                if self.weight_transport:
                    rnn.w_m = np.copy(rnn.w_out.T)
        
        """ Offline Update """
        # if not online, accumulate weight updates
        if not self.online:
            if 'w_out' in self.apply_to:
                self.dw_out += rnn.eta_out*np.outer(rnn.err, rnn.h)/t_max
            if 'w_rec' in self.apply_to: 
                self.dw_rec += rnn.eta_rec * np.outer(np.dot(rnn.w_m, rnn.err),np.ones(rnn.n_rec)) * self.p/t_max
            if 'w_in' in self.apply_to:
                self.dw_in += rnn.eta_in * np.outer(np.dot(rnn.w_m, rnn.err),np.ones(rnn.n_in)) * self.q/t_max
            if 'w_fb' in self.apply_to:
                self.dw_fb += rnn.eta_fb * np.outer(np.dot(rnn.w_m, rnn.err),np.ones(rnn.n_out)) * self.p_fb/t_max
                
        # if not online, add accumulated weight updates at the final step
        if not self.online and index == task.trial_duration-1:

                    
            if 'w_rec' in self.apply_to: 
                rnn.w_rec = rnn.w_rec + self.dw_rec
            if 'w_in' in self.apply_to:
                rnn.w_in = rnn.w_in + self.dw_in
            if 'w_fb' in self.apply_to:
                rnn.w_fb = rnn.w_fb + self.dw_fb
                
            # Note that w_out is updated at the end, so that w_m is updated _after_ all the other weights
            if 'w_out' in self.apply_to:
                rnn.w_out = rnn.w_out + self.dw_out
                
                # update w_m as well
                if self.weight_transport:
                    rnn.w_m = np.copy(rnn.w_out.T)
                    
            self.reset_learning_vars() # important for offline learning


    # TODO: define this function
    def reset_learning_vars(self):
        
        self.p = np.zeros((self.rnn.n_rec, self.rnn.n_rec)) # TODO: Check
        self.q = np.zeros((self.rnn.n_rec, self.rnn.n_in))
        self.p_fb = np.zeros((self.rnn.n_rec, self.rnn.n_out))
        
        self.dw_in = np.zeros((self.rnn.n_rec, self.rnn.n_in))
        self.dw_rec = np.zeros((self.rnn.n_rec, self.rnn.n_rec))
        self.dw_out = np.zeros((self.rnn.n_out, self.rnn.n_rec))
        self.dw_fb = np.zeros((self.rnn.n_rec, self.rnn.n_out))
        

                
                
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
        for k in ['apply_to', 'online', 'weight_transport']:
                print(k,': ',vars(self)[k])

            
            