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


class BPTT(LearningAlgorithm):
    
    """
    Backpropagation Through Time
    
    Backpropagation Through Time (BPTT) is a standard learning algorithm for RNNs. It is considered to be biologically implausible
    
    This implementation is adapted from Murray 2019 "Local online learning in recurrent networks with random feedback"
    (see https://github.com/murray-lab/rflo-learning for more details)
    """
    
    def __init__(self, rnn: RNN, apply_to: List[str]=['w_rec'], online: bool = False, weight_transport: bool = True) -> None:
        
        """
        Initialize learning rule and set which weights to apply to
        
        Args:
            rnn (RNN): RNN object
            apply_to (list): list of weights to apply learning rule, e.g. 'w_rec' or 'w_in'
            online (bool): whether learning rule is online (update every step) or offline (update at end of trial)
            weight_transpot (bool): if True, updare `w_m` every time `w_out` is updated. Although not normally used for BPTT, it allows for separating the effects of the readout weight transpose from the nonlocal rollout/update in time
            
        """
        print('>> TODO: Learning rule for w_fb not currently implemented for BPTT')
        print('>> TODO: implement weight_transport=False (of w_m) for BPTT')
        
        
        self.rnn = rnn
        
        # Initialize learning variables - note that these need to be reset at the end of each
        # trial with `reset_learning_vars()`
        self.dw_in = np.zeros((self.rnn.n_rec, self.rnn.n_in))
        self.dw_rec = np.zeros((self.rnn.n_rec, self.rnn.n_rec))
        self.dw_out = np.zeros((self.rnn.n_out, self.rnn.n_rec))
        self.dw_fb = np.zeros((self.rnn.n_rec, self.rnn.n_out))
        
        # TODO check that weight flags match weights in rnn
        assert apply_to != [], 'Must specify which weights to apply learning rule to with "apply_to"'
        
        assert not online, 'BPTT can only be implemented offline (i.e. updates occur at the end of the trials)'
        
        for a in apply_to:
            assert a in ['w_in','w_rec','w_out','w_fb'], "specified weights must be selected from ['w_in','w_rec','w_out','w_fb']"
            
        assert 'w_fb' not in apply_to, 'Learning rule for w_fb not currently implemented for BPTT'
            
        if 'w_in' in apply_to:
            assert rnn.eta_in, "eta_in must be specified if learning is occurring in w_in"
        if 'w_rec' in apply_to:
            assert rnn.eta_rec, "eta_rec must be specified if learning is occurring in w_rec"
        if 'w_out' in apply_to:
            assert rnn.eta_out, "eta_out must be specified if learning is occurring in w_out"
        if 'w_fb' in apply_to:
            assert rnn.eta_fb, "eta_fb must be specified if learning is occurring in w_fb"
            
        
        self.name = 'BPTT'
        self.apply_to = apply_to
        self.online = online
        self.weight_transport = weight_transport
        
        
        # Keep track of error, u, x_in and h across all timesteps in a trial
        self.err_history = [] #should eventually be size np.zeros((t_max, self.rnn.n_out))  
        self.u_history = []
        self.x_in_history = []
        self.h_history = []
        
        # TO DO: log error function
        
    
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
        
        # TODO: Allow for predefined teaching signal for error calculation
        #rnn.err = np.expand_dims(task.y_teaching_signal[index],1) - rnn.pos
        
        """ Error based on final target position """
        # scaled error based on time left in trial
        rnn.err = (1/(t_max-index)) * (task.y_target - rnn.pos)
        rnn.loss = 0.5*np.linalg.norm(rnn.err)**2
        
        self.err_history.append(np.copy(rnn.err)) # for BPTT, need to keep track of full history (not just 1 timestep)
        self.u_history.append(np.copy(rnn.u))
        self.x_in_history.append(np.copy(rnn.x_in))
        self.h_history.append(np.copy(rnn.h))
        
        # TODO: Alternative errors
        #rnn.err = (index/task.trial_duration) * (task.y_target - rnn.pos)/np.linalg.norm(task.y_target - rnn.pos)
        #rnn.err = (task.y_target - rnn.pos)
                
        # For BPTT, only update at end of the trial
        if index == t_max-1:
            
            # convert to np.array
            self.err_history = np.asarray(self.err_history).squeeze() # should be size np.zeros((t_max, self.rnn.n_out))
            self.x_in_history = np.asarray(self.x_in_history).squeeze()
            self.h_history = np.asarray(self.h_history).squeeze()
            self.u_history = np.asarray(self.u_history).squeeze()
            
            z = np.zeros((t_max, rnn.n_rec))
            
            # For standard backpropagation through time, the update should look like this:
            #
            # >> z[-1] = np.dot((rnn.w_out).T, self.err_history[-1])
            #
            # However, since we want to dissociate the matrix used in the forward pass from
            # the matrix used in the backward pass, we use rnn.w_m instead
            
            z[-1] = np.dot(rnn.w_m, self.err_history[-1]) # note w_m here
            
            # Loop backwards through timesteps
            for tt in range(t_max-1, 0, -1):
                z[tt-1] = z[tt]*(1 - rnn.dt/rnn.tau_rec)
                z[tt-1] += np.dot(rnn.w_m, self.err_history[tt]) # what are dimensions of rnn.err? It does not keep a history over timesteps!!
                z[tt-1] += np.dot(z[tt]*rnn.df(self.u_history[tt]), rnn.w_rec)*rnn.dt/rnn.tau_rec

                # Updates for the weights at each timestep backwards:
                if 'w_rec' in self.apply_to:
                    self.dw_rec += rnn.eta_rec/(t_max*rnn.tau_rec/rnn.dt)*np.outer(z[tt]*rnn.df(self.u_history[tt]),
                                                            self.h_history[tt-1])
                if 'w_in' in self.apply_to:
                    self.dw_in += rnn.eta_in/(t_max*rnn.tau_rec/rnn.dt)*np.outer(z[tt]*rnn.df(self.u_history[tt]),
                                                           self.x_in_history[tt])
                    
                if 'w_out' in self.apply_to:
                    self.dw_out += rnn.eta_out*np.outer(self.err_history[tt], self.h_history[tt])/t_max
                    
                     # update w_m as well (this is the standard BPTT algorithm)
                    if self.weight_transport:
                        rnn.w_m = np.copy(rnn.w_out.T)
                
                # TO DO: ADD RULE FOR W_FB
                
            # Update weights after unrolling in time
            if 'w_out' in self.apply_to:
                rnn.w_out = rnn.w_out + self.dw_out
            if 'w_rec' in self.apply_to: 
                rnn.w_rec = rnn.w_rec + self.dw_rec
            if 'w_in' in self.apply_to:
                rnn.w_in = rnn.w_in + self.dw_in
            
            #if 'w_fb' in self.apply_to:
            #    rnn.w_fb = rnn.w_fb + self.dw_fb
                
            self.reset_learning_vars()

            
            
            
    def reset_learning_vars(self):
                
        self.err_history = [] 
        self.u_history = []
        self.x_in_history = []
        self.h_history = []
        
        self.dw_in = np.zeros((self.rnn.n_rec, self.rnn.n_in))
        self.dw_rec = np.zeros((self.rnn.n_rec, self.rnn.n_rec))
        self.dw_out = np.zeros((self.rnn.n_out, self.rnn.n_rec))
        self.dw_fb = np.zeros((self.rnn.n_rec, self.rnn.n_out))
        
        
        
        
    def print_params(self) -> None:
        
        """ Print Hyperparameters """
        for k in ['apply_to', 'online']:
                print(k,': ',vars(self)[k])
            
            