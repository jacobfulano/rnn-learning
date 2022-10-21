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


class TrackVars(LearningAlgorithm):
    
    """
    This is not a learning algorithm, it simply keeps track of variables such as 
    error (MSE) and loss
    
    TODO: there is probably a cleaner way to do this
    TODO: Track other definitions of variables/errors
    """
    
    def __init__(self, rnn: RNN, apply_to: List[str]=[]) -> None:
        
        """
        Initialize learning rule and set which weights to apply to
        

        """
        
        self.rnn = rnn

        assert apply_to == [], '"apply_to" Must be empty'
        
        
        self.name='TrackVars'
        self.apply_to = apply_to
        
        # TO DO: log error function
        
    
    def update_learning_vars(self, index: int, task: Task):
        
        """
        Update variables associated with learning
        
        Args:
            index (int): trial step
            task (Task): task object that specifies targets, trial duration, etc.
        """
        
        # pointer for convenience
        rnn = self.rnn
        t_max = task.trial_duration
        
        # TODO: Allow for predefined teaching signal for error calculation
        #rnn.err = np.expand_dims(task.y_teaching_signal[index],1) - rnn.pos
        
        """ Error based on final target position """
        # scaled error based on time left in trial
        rnn.err = (1/(task.trial_duration-index)) * (task.y_target - rnn.pos)
        rnn.loss = 0.5*np.linalg.norm(rnn.err)**2
        
        # TODO: Alternative errors
        #rnn.err = (index/task.trial_duration) * (task.y_target - rnn.pos)/np.linalg.norm(task.y_target - rnn.pos)
        #rnn.err = (task.y_target - rnn.pos)

        
        
    # TODO: define this function
    def reset_learning_vars(self):
        
        pass
        
        
    def print_params(self) -> None:
        
        """ Print Hyperparameters """
        for k in ['apply_to']:
                print(k,': ',vars(self)[k])
                

            
            