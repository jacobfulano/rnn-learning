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


class WeightMirror(LearningAlgorithm):
    
    """
    Weight Mirroring
    
    Weight mirroring simply injects some noise at an individual layer in the forward pass and uses the correlation of the noisy output and the noise itself to update the "backward pass"/"transpose" matrix.
    
    See "Deep Learning without Weight Transport" (Akrout et al. 2019) for more details
    """
    
    def __init__(self, rnn: RNN, apply_to: List[str]=['w_m'], online: bool = False) -> None:
        
        """
        Initialize learning rule and set which weights to apply to
        
        Args:
            rnn (RNN): RNN object
            apply_to (list): list of weights to apply learning rule, e.g. 'w_rec' or 'w_in'
            online (bool): whether learning rule is online (update every step) or offline (update at end of trial)
            
        Variables used to keep track of derivatives:
            dw_m: update to internal model weights
        """
        
        # Initialize learning variables
        self.rnn = rnn
        
        self.dw_m = np.zeros((self.rnn.n_rec, self.rnn.n_out)) # transpose of w_out
        
        assert apply_to != [], 'Must specify which weights to apply learning rule to with "apply_to"'
        
        # check that weight flags match weights in rnn
        for a in apply_to:
            assert a in ['w_m'], "specified weights must be selected from ['w_m']"
            
        if 'w_m' in apply_to:
            assert rnn.sig_m, "sigma_m must be specified if learning is occurring in w_m"
            assert rnn.eta_m, "eta_m must be specified if learning is occurring in w_m"
            assert rnn.lam_m, "lambda_m must be specified if learning is occurring in w_m"
        
        self.name='WeightMirror'
        self.apply_to = apply_to
        self.online = online        
                
    
    def update_learning_vars(self, index: int, task: Task):
        
        """
        Update variables associated with learning
        
        Args:
            index (int): trial step
            task (Task): task object that specifies targets, trial duration, etc.
        
        Variables use to keep track of derivatives
            dw_m: change internal model of credit assignment
            
        """
        
        # pointer for convenience
        rnn = self.rnn
        t_max = task.trial_duration
        
        # TODO: Allow for predefined teaching signal for error calculation
        #rnn.err = np.expand_dims(task.y_teaching_signal[index],1) - rnn.pos
        
        """ Error based on final target position """
        # scaled error based on time left in trial. Store for possible probes
        rnn.err = (1/(task.trial_duration-index)) * (task.y_target - rnn.pos)
        rnn.loss = 0.5*np.linalg.norm(rnn.err)**2
        
        """ Weight Mirroring """    
        self.hh = rnn.sig_m*rnn.rng.randn(rnn.n_rec)
        self.yy = np.dot(rnn.w_out, self.hh)
                

        """ Online Update """
        if self.online:
                    
            if 'w_m' in self.apply_to:
                self.dw_m = rnn.eta_m * np.outer(self.hh,self.yy) - rnn.lam_m * rnn.w_m
                rnn.w_m = rnn.w_m + self.dw_m
                
        
        """ Offline Update """
        # if not online, accumulate weight updates
        if not self.online:
            
            if 'w_m' in self.apply_to:
                self.dw_m += rnn.eta_m * np.outer(self.hh,self.yy) - rnn.lam_m * rnn.w_m # note difference with 


        # if not online, add accumulated weight updates at the final step
        if not self.online and index == task.trial_duration-1:

            if 'w_m' in self.apply_to: 
                rnn.w_m = rnn.w_m + self.dw_m
                    
            self.reset_learning_vars() # important for offline learning


    # TODO: define this function
    def reset_learning_vars(self):
        
        self.dw_m = np.zeros((self.rnn.n_rec, self.rnn.n_out))
        
        
        
    def print_params(self) -> None:
        
        """ Print Hyperparameters """
        for k in ['apply_to', 'online', 'weight_transport']:
                print(k,': ',vars(self)[k])
                

            
            