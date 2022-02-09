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
            
    TODO: Currently only implemented for w_rec. Need to implement for w_in, w_fb and w_out
    
    TODO: SOMETHING IS CURRENTLY WRONG WITH MULTI-TASK TRAINING
    """
    
    def __init__(self, rnn: RNN, sig_xi: float, tau_reward: float, apply_to: List[str]=['w_rec'], online: bool = True) -> None:
        
        
        # variable necessary for this RL algorithm
        self.sig_xi = sig_xi # noise scale
        self.tau_reward = tau_reward # timescale of reward
        
        # Initialize learning variables
        self.rnn = rnn
        self.p = np.zeros((self.rnn.n_rec, self.rnn.n_rec))
        
        self.r_av = [] # should be a vector dependent on the task
        self.r_av_prev = [] # should be a vector dependent on the task
        self.rnn.r_current = 0
        
        self.all_tasks = [] # keeps track of unique tasks (when training with multiple tasks)
        self.task_idx = None
        
        self.bonus = 0
        
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
            raise Exception("REINFORCE does not currently work for w_fb")
        #    assert rnn.eta_fb, "eta_fb must be specified if learning is occurring in w_fb"
                
        self.apply_to = apply_to
        self.online = online
                
        assert apply_to[0] == 'w_rec', 'REINFORCE only currently implemented for w_rec' 
        
    
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
            dw_fb: change in feedback weights
        """
        
        """ Keep separate running average for each task """
        #if index == 0: # this only needs to be calculated at the beginning of the trial
        self.task_idx = self._track_tasks(task)
        
        task_idx = self.task_idx
                    
        
        # pointer for convenience
        rnn = self.rnn
        t_max = task.trial_duration

            
        """ Reward based on final target position """
        #xi = self.sig_xi * rnn.rng.randn(rnn.n_rec,1) # note rand, and vector of noise values
        xi = self.sig_xi * rnn.rng.randn(1) * np.ones((rnn.n_rec,1)) # only one noise value
                
        self.p = (1-1/rnn.tau_rec)*self.p
        self.p += np.outer(xi*df(rnn.u), rnn.h_prev)/rnn.tau_rec

        # BONUS
#         if index > task.trial_duration-np.round(task.trial_duration/4): # end of trial
#             #norm_av += np.linalg.norm(pos[tt+1]-y_)**2
#             if  np.linalg.norm(rnn.pos-task.y_target)**2 < 0.5: #0.35: # doesn't happen often?
#                 #bonus = bonus_amount
#                 self.bonus += 1
                
                
                    
        if self.online:
            
            rnn.r_current = -(np.linalg.norm(task.y_target-rnn.pos))**2 + self.bonus
            
            dw_rec = rnn.eta_rec * (rnn.r_current - self.r_av[task_idx])*self.p/t_max
            
            if 'w_rec' in self.apply_to: 
                rnn.w_rec = rnn.w_rec + dw_rec
                
        # TO DO: ADD FOR w_in, w_out and w_fb
                
        """ At the end of the trial """
        if not self.online and index == task.trial_duration-1:
            
            #print(task_idx)
            #print(task.y_target)
            #print('first')
            
            rnn.r_current = -(np.linalg.norm(task.y_target - rnn.pos))**2 + self.bonus
            
            #print('y_target',task.y_target)
            #print('rnn.pos',rnn.pos)
            #print('task.y_target - rnn.pos',task.y_target - rnn.pos)
            #print('norm',-np.linalg.norm(task.y_target - rnn.pos)**2)
            
            dw_rec = rnn.eta_rec * (rnn.r_current - self.r_av[task_idx])*self.p
            
            #print('rnn.r_current - self.r_av[task_idx] = ',rnn.r_current - self.r_av[task_idx])
            #print('dw_rec',dw_rec)
            #print('sum dw_rec:',np.sum(dw_rec))
    
            if 'w_rec' in self.apply_to: 
                rnn.w_rec = rnn.w_rec + dw_rec
        
        """ At end of trial, update average reward"""
        # TODO: This needs to be implemented for multiple targets
        if index == task.trial_duration-1:
            #print(task_idx)
            
            #self.r_av_prev[task_idx] = np.copy(self.r_av[task_idx]) # this leads to an array inside a list?
            self.r_av_prev[task_idx] = self.r_av[task_idx]
            self.r_av[task_idx] = self.r_av_prev[task_idx] + (1/self.tau_reward) * (rnn.r_current-self.r_av_prev[task_idx])
            #print(self.r_av)
            #print('r_av_prev',self.r_av_prev)
            #print('r_av',self.r_av)
            #print('task idx: ',task_idx)
            #print('r_av',self.r_av)
            #print()
            
            

    def reset_learning_vars(self):
        
        """ Reset variables """
        
        self.p = np.zeros((self.rnn.n_rec, self.rnn.n_rec))
        self.rnn.r_av = []
        self.rnn.r_av_prev = []
        self.rnn.r_current = 0
     
    
    def _track_tasks(self,task):
            
        """ Keep track of which task is being used for training """
        
        task_bool = [np.array_equal(ts.y_target,task.y_target) for ts in self.all_tasks]

        if np.any(task_bool):
            task_idx = np.argwhere(task_bool).squeeze()

        else:
            self.all_tasks.append(task)
            self.r_av.append(0)
            self.r_av_prev.append(0)
            
            task_idx = len(self.all_tasks)-1
            
                    
        return task_idx
    