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

            
class REINFORCE1D(LearningAlgorithm):
    
    """
    REINFORCE 
    
    This is a reinforcement learning (RL) rule based on the paper "Biologically plausible learning in 
    recurrent neural networks reproduces neural dynamics observed during cognitive tasks" (Miconi 2017)
    
    Args:
        rnn (RNN): RNN object
        sig_xi (float): noise scale used in learning rule
        tau_reward (float): timescale of reward memory
        apply_to (list): list of weights to apply learning rule, e.g. 'w_rec' or 'w_in'
        online (bool): whether learning rule is online (update every step) or offline (update at end of trial)
            
    Variables used to keep track of derivatives:
        p: eligibility trace for the recurrent weights
            
    TODO: Currently only implemented for w_rec. Need to implement for w_in, w_fb and w_out
    """
    
    def __init__(self, rnn: RNN,  tau_reward: float, apply_to: List[str]=['w_rec'], online: bool = True) -> None:
        
        #sig_xi: float,
        
        # variablce necessary for this RL algorithms
        #self.sig_xi = sig_xi # noise scale
        self.tau_reward = tau_reward # timescale of reward
        
        # Initialize learning variables
        self.rnn = rnn
        self.p = np.zeros((self.rnn.n_rec, self.rnn.n_rec))
        self.dw_rec = np.zeros((self.rnn.n_rec, self.rnn.n_rec))
        
        self.r_av = 0 # should be a vector dependent on the task
        self.r_av_prev = 0 # should be a vector dependent on the task
        self.rnn.r_current = 0
        
        # TODO check that weight flags match weights in rnn
        assert apply_to, 'Must specify which weights to apply learning rule to with "apply_to"'
        self.apply_to = apply_to
        self.online = online
                
        assert apply_to[0] == 'w_rec', 'REINFORCE only currently implemented for w_rec' 
        
    
    def update_learning_vars(self, index: int, task: Task):
        
        """
        Update variables associated with learning
        
        Args:
            index (int): trial step
            task (Task): task object that specifies targets, trial duration, etc.
        
        Variables use to keep track of derivatives
            dw_rec: change in the recurrent weights
            
        TODO: Implement for
            dw_out: change in the output weights
            dw_in: change in input weights
            dw_fb: change in feedback weights
        """
        
        # pointer for convenience
        rnn = self.rnn
        t_max = task.trial_duration

            
        """ Reward based on final target position """
        #xi = self.sig_xi * rnn.rng.randn(rnn.n_rec,1)
                
        self.p = (1-1/rnn.tau_rec)*self.p
        self.p += np.outer(rnn.xi*df(rnn.u), rnn.h_prev)/rnn.tau_rec

            
        rnn.r_current = -(np.linalg.norm(task.y_target-rnn.pos))**2
            
        
        
        if self.online:
            
            if 'w_rec' in self.apply_to: 
                self.dw_rec = rnn.eta_rec * (rnn.r_current - self.r_av)*self.p/t_max
                """ Update Weights """
                rnn.w_rec = rnn.w_rec + self.dw_rec

                
        if not self.online:
            
            if 'w_rec' in self.apply_to: # keep running sum of update
                self.dw_rec += rnn.eta_rec * (rnn.r_current - self.r_av)*self.p/t_max
                
        """ At the end of the trial """
        if not self.online and index == task.trial_duration-1:
            
            # Sporadic reward at end of trial
            #rnn.r_current = -(np.linalg.norm(task.y_target - rnn.pos))**2
            #dw_rec = rnn.eta_rec * (rnn.r_current - self.r_av)*self.p
    
            if 'w_rec' in self.apply_to: 
                rnn.w_rec = rnn.w_rec + self.dw_rec
                
        
        """ At end of trial, update average reward"""
        # TODO: This needs to be implemented for multiple targets
        if index == task.trial_duration-1:
            self.r_av = self.r_av_prev + (1/self.tau_reward) * (rnn.r_current-self.r_av_prev)
            self.r_av_prev = np.copy(self.r_av)
            
            self.reset_learning_vars() # important for offline learning
            
            

    def reset_learning_vars(self):
        
        """ Reset variables """
        self.dw_rec = np.zeros((self.rnn.n_rec, self.rnn.n_rec))
        
        self.p = np.zeros((self.rnn.n_rec, self.rnn.n_rec))
        #self.rnn.r_av = 0
        #self.rnn.r_av_prev = 0
        self.rnn.r_current = 0