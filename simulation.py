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
from rnn import RNNparams, RNN
from utils.plotting import plot_position
from task import Task
from algorithms.base import LearningAlgorithm

from utils.functions import rgetattr


class Simulation():
    
    """
    Run Simulations with trial and session structure
    
    The general structure of this class, and probes/monitors in particular, are directly inspired 
    from Owen Marschall https://github.com/omarschall/vanilla-rtrl/
    
    Args:
        rnn (RNN):
    """
    
    def __init__(self,rnn: RNN) -> None:
        
        self.rnn = rnn
        
        
    def run_session(self,n_trials: int, tasks: List[Task], learn_alg: List[str], probe_types: List[str], plot: bool = True, plot_freq: int = 10) -> None:
        
        """ Shuffle Indices for tasks """
        idxs = self.rnn.rng.choice(np.arange(0,len(tasks)), size=n_trials) # shuffle presentation of stimuli
        
        if plot:
            fig = plt.figure(figsize=(6,5))
            assert 'pos' in probe_types, "In order to plot position, must include 'pos' in probe_types"
        
        for count,idx in enumerate(idxs):
            
            self.run_trial(tasks[idx],learn_alg=learn_alg,probe_types=probe_types,train=True)
            
            if plot and count % plot_freq == 0:
                
                #plt.plot(self.probes['pos'].squeeze()[:,0],self.probes['pos'].squeeze()[:,1],label='trial {}'.format(count))
                fig = plot_position(fig=fig, pos=self.probes['pos'], tasks = tasks, count=count, n_trials=n_trials, plot_freq=plot_freq)
    
        
    
    def run_trial(self, task: Task, 
                  train: bool=True, 
                  learn_alg: List[LearningAlgorithm]=[], 
                  probe_types: List[str]=[]) -> None:
        """
        Run Trial
        
        Run forward as many timesteps as necessary, in either train or test mode
        depending on the learning rule, this can run through full trial
        
        Length of the trial is specified by Task
        
        Arguments
            task (Task): task object that contains details of target, trial duration, etc.
            train (bool): whether in training mode
            learn_alg (List[LearningAlgorithm]): , default = []
            probe_types (list): 
        """
        
        assert self.rnn.n_in == task.x_in.shape[1], 'Task non temporal input must match RNN input dimensions'
        
        if train and not learn_alg:
            raise AssertionError('If training, need to specify learning algorithm')
            
        self.learn_alg = learn_alg
        
        # initialize probes
        self.probe_types = probe_types
        self.probes = {probe:[] for probe in self.probe_types}
        
        """ Begin Trial """
        for tt in range(task.trial_duration): # note this used to be trial_duration-1
            
            self.forward_step(task.x_in[tt]) # all it gets is external input 
            
            if train:
                self.train_step(tt,train,task)
        
            self.update_probes()
            
        self.probes_to_arrays()
        
        self.reset_trial()
        
    
    def forward_step(self, x) -> None:
        """ run network forward one step"""
        
        # pointer for convenience
        rnn = self.rnn

        # run network forward one step and get predictions
        rnn.next_state(np.expand_dims(x,1))
        rnn.output()

        
    def train_step(self,index: int, train: bool, task: Task):
        
        """ Apply Training Step 
        
        Note that this can apply multiple learning rules to multiple matrices.
        It is incumbent on the user to ensure that there are no conflicts between learning rules
        
        Arguments
            index (int):
            train (bool):
            task (Task):
        """
        
        for learn_alg in self.learn_alg:
            learn_alg.update_learning_vars(index,task)
        
        
    def update_probes(self):
        """Loops through the probe keys and appends current value of any
        object's attribute found."""

        for key in self.probes:
            try:
                self.probes[key].append(rgetattr(self.rnn, key))
                #print('>>',key,rgetattr(self.rnn, key))
            except AttributeError:
                pass
            
    def probes_to_arrays(self):
        """Recasts monitors (lists by default) as numpy arrays for ease of use
        after running."""

        for key in self.probes:
            try:
                self.probes[key] = np.array(self.probes[key])
            except ValueError:
                pass
        
    def reset_trial(self):
        
        """ Reset some parameters 
        
        This is particularly important at the end of a trial
        """
        
        # pointer for convenience
        rnn = self.rnn
    
        
        rnn.x_in = 0
        rnn.h0 = np.zeros((rnn.n_rec,1))
        rnn.h = np.copy(rnn.h0)
        rnn.y_out = np.zeros((rnn.n_out,1))
        rnn.pos = np.zeros((rnn.n_out,1))
        
        if rnn.velocity_transform:
            rnn.vel = np.zeros((rnn.n_out,1))
        else:
            rnn.vel = None
        
        
