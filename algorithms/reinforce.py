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

from algorithms.base import LearningAlgorithm
from rnn import RNNparams, RNN
from task import Task

from utils.functions import f, df, theta, rgetattr

            
class REINFORCE(LearningAlgorithm):
    
    """
    REINFORCE (Miconi 2017)
    
    
    """
    
    def __init__(self, rnn: RNN, sig_xi: float, tau_reward: float, apply_to: List[str]=['w_rec'], online: bool = True) -> None:
        
        """ reinforcement learning """
        self.sig_xi = sig_xi
        self.tau_reward = tau_reward
    
        
        # Initialize learning variables
        self.rnn = rnn
        self.p = np.zeros((self.rnn.n_rec, self.rnn.n_rec))
        
        self.r_av = 0 # should be a vector dependent on the task
        self.r_av_prev = 0 # should be a vector dependent on the task
        self.rnn.r_current = 0
        
        # check that weight flags match weights in rnn
        # TODO
        assert apply_to, 'Must specify which weights to apply learning rule to with "apply_to"'
        self.apply_to = apply_to
        self.online = online
                
        assert apply_to[0] == 'w_rec', 'REINFORCE only currently implemented for w_rec' 
        
    
    def update_learning_vars(self, index: int, task: Task):
        
        """
        Update variables associated with learning
        
        Args:
            p:
            q:
            dw_out:
            dw_rec:
            dw_in:
        """
        
        # pointer for convenience
        rnn = self.rnn
        t_max = task.trial_duration
        
        # predefined teaching signal
        #rnn.err = np.expand_dims(task.y_teaching_signal[index],1) - rnn.pos
        
        
            
        """ Reward based on final target position """
        # don't need this...
        #rnn.err = (index/task.trial_duration) * (task.y_target - rnn.pos)/np.linalg.norm(task.y_target - rnn.pos)
        
        xi = self.sig_xi * rnn.rng.randn(rnn.n_rec,1)
                
        self.p = (1-1/rnn.tau_rec)*self.p
        self.p += np.outer(xi*df(rnn.u), rnn.h_prev)/rnn.tau_rec

            
        if self.online:
            
            rnn.r_current = -(np.linalg.norm(task.y_target-rnn.pos))**2
            
            dw_rec = rnn.eta_rec * (rnn.r_current - self.r_av)*self.p/t_max
            
            if 'w_rec' in self.apply_to: 
                rnn.w_rec = rnn.w_rec + dw_rec
                
        """ at the end of the trial """
        if not self.online and index == task.trial_duration-1:
            
            rnn.r_current = -(np.linalg.norm(task.y_target - rnn.pos))**2
            
            dw_rec = rnn.eta_rec * (rnn.r_current - self.r_av)*self.p
    
            if 'w_rec' in self.apply_to: 
                rnn.w_rec = rnn.w_rec + dw_rec
        
        """ at end of trial update average reward"""
        if index == task.trial_duration-1:
            self.r_av = self.r_av_prev + (1/self.tau_reward) * (rnn.r_current-self.r_av_prev)
            self.r_av_prev = np.copy(self.r_av)
            
            

    def reset_learning_vars(self):
        
        self.p = np.zeros((self.rnn.n_rec, self.rnn.n_rec))
        self.rnn.r_av = 0
        self.rnn.r_av_prev = 0
        self.rnn.r_current = 0
    