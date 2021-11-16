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

from utils.functions import f, df, theta, rgetattr


@dataclass
class RNNparams():
    
    """
    Hyperparameter class for RNN
    
    Attributes:
        n_in (int): dimension of inputs
        n_rec (int): dimension of recurrent units
        n_out (int): dimension of output
        tau_rec (float): time constant for recurrent weights
        tau_vel (float): time constant for velocity transformation
        tau_reward (float): time constant for reward (RL)
        driving_feedback (bool): whether there is driving feedback
        velocity_transform (bool): whether to apply low pass filter to neural readout
        rng: random number generator
    """
    
    """ number of units at each layer """
    n_in: int
    n_rec: int
    n_out: int
        
    """ noise """
    sig_in: float
    sig_rec: float
    sig_out: float

    tau_rec: float
        
    """ learning rates for each population """
    # note that this does not mean
    eta_in: Optional[float] = None
    eta_rec: Optional[float] = None
    eta_out: Optional[float] = None
    
    driving_feedback: bool = False
    eta_fb: Optional[float] = None # learning rate for feedback weights
    feedback_signal: Optional[str] = 'position'
    
    """ velocity transform """
    velocity_transform: bool = False
    tau_vel: Optional[float] = None
    dt_vel: Optional[float] = None
        
    rng: np.random.RandomState() = np.random.RandomState(17)



class RNN():
    
    """
    RNN class
    
    Initialize Weights
    
    Perform surgery and insert/set weights
    
    Advance to next RNN state
    
    Arguments
        params (RNNparams): dataclass object that stores hyperparameters for network
        init (boolean): if true, initialize weights. default=True
    """
    
    def __init__(self, params: RNNparams,init=True):
        for key, value in dataclasses.asdict(params).items():
            setattr(self, key, value)
        
        if init:
            self.initialize_weights()
        
        # Initialize
        self.x_in = 0
        self.h0 = np.zeros((self.n_rec,1)) # initial activity of the RNN
        self.h = np.copy(self.h0)
        self.y_out = np.zeros((self.n_out,1))
        self.pos = np.zeros((self.n_out,1))
        self.u = np.zeros((self.n_rec,1))
        
        if self.velocity_transform:
            self.vel = np.zeros((self.n_out,1))
            
            assert self.dt_vel, "If applying a velocity transform, dt_vel must be specified"
        else:
            self.vel = None
            
        if self.driving_feedback:
            assert self.eta_fb is not None, "If driving feebdack, eta_fb must be set"
            assert self.feedback_signal in ['position','error'], "Must specify if feedback_signal from {'position','error'}"
            
        # TO DO I don't want this to be here, but think it is necessary for probes?
        self.r = None
        self.r_current = None
        self.err = np.zeros((self.n_out,1))
        
        
         
    def initialize_weights(self) -> None:
        
        """ Initialize weights with random number generator """
    
        self.w_in = 2*(self.rng.rand(self.n_rec, self.n_in) - 1) # changed from 0.1
        self.w_rec = 1.5*self.rng.randn(self.n_rec, self.n_rec)/self.n_rec**0.5 # --> 1 changed from 1.5 # why randn instead of rand?
        self.w_out = 2*(2*self.rng.rand(self.n_out, self.n_rec) - 1)/self.n_rec**0.5 
        
        self.w_m = np.copy(self.w_out).T # CHANGE THIS
        
        if self.driving_feedback:
            self.w_fb = 2*self.rng.randn(self.n_rec,self.n_out)/self.n_rec**0.5

    def set_weights(self, 
                    w_in: Optional[np.array]=None, 
                    w_rec: Optional[np.array]=None, 
                    w_out: Optional[np.array]=None, 
                    w_m: Optional[np.array]=None,
                    w_fb: Optional[np.array]=None):
        
        """ Set weights with predefined values 
        
        Args:
            w_in: input matrix
            w_rec: recurrent matrix
            w_out: output matrix
            w_m: "transpose" matrix that updates learning (in SL)
            w_fb: feedback matrix that drives RNN activity
        """
        
        if w_in is not None:
            assert w_in.shape == (self.n_rec,self.n_in), 'Dimensions must be (n_rec,n_in)'
            self.w_in = w_in # = w_init['w_in'] #0.1*(np.random.rand(n_rec, n_in) - 1)
        if w_rec is not None:
            assert w_rec.shape == (self.n_rec,self.n_rec), 'Dimensions must be (n_rec,n_rec)'
            self.w_rec = w_rec # = w_init['w_rec'] #1*np.random.randn(n_rec, n_rec)/n_rec**0.5 # --> changed from 1.5
        if w_out is not None:
            assert w_out.shape == (self.n_rec,self.n_out), 'Dimensions must be (n_rec,n_out)'
            self.w_out = w_out # = w_init['w_out'] #1*(2*np.random.rand(n_out, n_rec) - 1)/n_rec**0.5 # --> should be on same scale as target
        if w_m is not None:
            assert w_m.shape == self.w_out.T.shape, 'Dimensions must be (n_out,n_rec)'
            self.w_m = w_m
        
        if w_fb is not None:
            assert self.driving_feedback, 'driving_feedback should be set to True'
            assert w_fb.shape == (self.n_rec,self.n_out), 'Dimensions must be (n_rec,n_out)'
            self.w_fb = w_fb
    
    # needs to specify 
    def next_state(self, x_in: np.array): # could be array
        """ Advance the network forward by one step 
        
        Args
        x_in: external input
        
        """
        
        self.h_prev = np.copy(self.h)
        self.x_in_prev = self.x_in
        
        """ recurrent activity """                        
        if self.driving_feedback:
            # note that feedback is position, which seems to do better than error
            if self.feedback_signal == 'position':
                self.u = np.dot(self.w_rec, self.h) + np.dot(self.w_in, x_in) + np.dot(self.w_fb, self.pos) + self.sig_in*self.rng.randn(self.n_in,1)
            
            # note that the feedback here is error, not position
            if self.feedback_signal == 'error':
                self.u = np.dot(self.w_rec, self.h) + np.dot(self.w_in, x_in) + np.dot(self.w_fb, self.err) + self.sig_in*self.rng.randn(self.n_in,1)

        else:
            self.u = np.dot(self.w_rec, self.h) + np.dot(self.w_in, x_in + self.sig_in*self.rng.randn(self.n_in,1))

        # updated step
        self.h = self.h + (-self.h + f(self.u) + self.sig_rec*self.rng.randn(self.n_rec,1))/self.tau_rec
        
        self.x_in = x_in
        #sensory_feedback = np.dot(self.w_fb,pos[tt]-y_ + self.sig_fb*self.noise_rng.randn(len(y[tt]))) 

       

    def output(self) -> None:
        
        """ velocity transform """
        
        self.y_prev = np.copy(self.y_out)
        
        # output
        self.y_out = np.dot(self.w_out, self.h) + self.sig_out*self.rng.randn(self.n_out,1)

        if self.velocity_transform:
            # cursor velocity
            self.vel = (1-1/self.tau_vel)*self.vel +  (1/self.tau_vel)*self.y_out

            # cursor position
            self.pos = self.pos + self.vel*self.dt_vel
            
        else:
            self.pos = self.y_out



