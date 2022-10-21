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
    Hyperparameter class for vanilla RNN
    
    Attributes:
        n_in (int): dimension of inputs
        n_rec (int): dimension of recurrent units
        n_out (int): dimension of output
        
        sig_in (float): noise scale of input weights
        sig_rec (float): noise scale of recurrent weights
        sig_out (float): noise scale of output weights
    
        tau_rec (float): time constant for recurrent weights
        
        eta_in (float): learning rate for input weights
        eta_rec (float): learning rate for recurrent weights
        eta_out (float): learning rate for output weights
                
        driving_feedback (bool): whether there is driving feedback
        eta_fb (float): learning rate for feedback weights
        feedback_signal (str):
    
        velocity_transform (bool): whether to apply low pass filter to neural readout
        tau_vel (float): time constant for velocity transformation
        dt_vel (float):
        
        rng: random number generator
        
    TODO: Feedback noise
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
    
    """ integration timescale of simulation 
    
    Note that if this value is changed to something other than 1, it affects
    the simulation in 2 locations:
    - the recurrent activity update, which is scaled by dt/tau_rec
    - the recurrent noise xi, which is scaled by sqrt(dt)
    """
    dt: Optional[float] = 1.0
        
    """ learning rates for each population """
    # note that this does not mean that the RNN necessarily learns
    eta_in: Optional[float] = None
    eta_rec: Optional[float] = None
    eta_out: Optional[float] = None
        
    g_in: Optional[float] = 2.0
    g_rec: Optional[float] = 1.5
    g_out: Optional[float] = 2.0
    g_fb: Optional[float] = 2.0
    
    """ driving feedback parameters """
    driving_feedback: bool = False
    eta_fb: Optional[float] = None # learning rate for feedback weights
    sig_fb: Optional[float] = None
    feedback_signal: Optional[str] = 'position'
    
    """ velocity transform """
    velocity_transform: bool = False
    tau_vel: Optional[float] = None
    dt_vel: Optional[float] = None
        
    """ recurrent noise dimension parameters """
    sig_rec_dim: Optional[int] = None
        
    """ weight mirroring parameters """
    eta_m: Optional[float] = None
    sig_m: Optional[float] = None
    lam_m: Optional[float] = None
        
    rng: np.random.RandomState() = np.random.RandomState(17)
        
    def print_params(self) -> None:
        
        """ Method to print hyperparameters """
        for k,v in dataclasses.asdict(self).items():
            print(k+':',v)



class RNN():
    
    """
    RNN class
    
    This class also holds the current state of the RNN, and the function
    'next_state' advances the RNN to the next state.
    
    Class functions can also initialize weights and insert/set weights.
    
    Args:
        params (RNNparams): dataclass object that stores hyperparameters for network
        init (boolean): if true, initialize weights. default=True
    """
    
    def __init__(self, params: RNNparams,init=True, f=f, df=df, sig_rec_covariance=None) -> None:
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
        
        self.f = f
        self.df = df
        
        if self.velocity_transform:
            self.vel = np.zeros((self.n_out,1))
            assert self.dt_vel, "If applying a velocity transform, dt_vel must be specified"
        else:
            self.vel = None
            
        if self.driving_feedback:
            assert self.eta_fb is not None, "If driving feebdack, eta_fb must be set"
            assert self.feedback_signal in ['position','error'], "Must specify if feedback_signal from {'position','error'}"
            assert self.sig_fb is not None, "If driving feedback, sig_fb must be set"
            
        # TO DO: I don't want this to be here, but think it is necessary for probes
        self.r = None
        self.r_current = None
        self.err = np.zeros((self.n_out,1)) # does this have to be here?
        
        """ Properties of recurrent noise """
        if self.sig_rec_dim == None:
            self.sig_rec_dim = self.n_rec # dimension of recurrent noise
            
        assert self.sig_rec_dim <= self.n_rec, 'recurrent noise dimension must be less than or equal to number of recurrent units'
            
        """ generate covariance matrix for recurrent noise
        this is used for sampling a multivariate gaussian via rng.multivariate_normal(mean,cov)"""
        
        # scenario1 - full rank, isotropic noise
        if sig_rec_covariance is None and self.sig_rec_dim == self.n_rec:
            self.sig_rec_covariance = self.sig_rec * np.eye(self.n_rec) # isotropic sample # note that this is sig_rec and not sig_rec**2
            
        # scenario 2 - low-D, isotropic noise
        elif sig_rec_covariance is None and self.sig_rec_dim < self.n_rec:
            C = self.sig_rec * np.eye(self.n_rec) # note that this is sig_rec and not sig_rec**2
            ind = self.rng.choice(np.arange(self.sig_rec_dim,dtype=int),1)
            C[ind] = 0 # set some neurons to zero
            self.sig_rec_covariance = C
        
        # scenario 3 - noise is specified by covariance matrix
        else:
            assert sig_rec_covariance.shape[0] == self.n_rec, 'covariance matrix must have shape (n_rec,n_rec)'
            assert sig_rec_covariance.shape[0] == self.n_rec, 'covariance matrix must have shape (n_rec,n_rec)'
            
            # CHECK - NOTE THERE IS NO MULTIPLICATION BY self.sig_rec HERE
            
            self.sig_rec_covariance = sig_rec_covariance

            
        
        
         
    def initialize_weights(self) -> None:
        
        """ Initialize all weights with random number generator """
    
        self.w_in = self.g_in*(self.rng.rand(self.n_rec, self.n_in) - 1) # changed from 0.1
        self.w_rec = self.g_rec*self.rng.randn(self.n_rec, self.n_rec)/self.n_rec**0.5
        self.w_out = self.g_out*(2*self.rng.rand(self.n_out, self.n_rec) - 1)/self.n_rec**0.5 
        
        self.w_m = np.copy(self.w_out).T # CHANGE THIS
        
        if self.driving_feedback:
            self.w_fb = self.g_fb*self.rng.randn(self.n_rec,self.n_out)/self.n_rec**0.5

    def set_weights(self, 
                    w_in: Optional[np.array]=None, 
                    w_rec: Optional[np.array]=None, 
                    w_out: Optional[np.array]=None, 
                    w_m: Optional[np.array]=None,
                    w_fb: Optional[np.array]=None) -> None:
        
        """ Set weights with predefined values 
        
        The weight(s) to set must be specified
        
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
            assert w_out.shape == (self.n_out,self.n_rec), 'Dimensions must be (n_out,n_rec)'
            self.w_out = w_out # = w_init['w_out'] #1*(2*np.random.rand(n_out, n_rec) - 1)/n_rec**0.5 # --> should be on same scale as target
        if w_m is not None:
            assert w_m.shape == self.w_out.T.shape, 'Dimensions must be (n_out,n_rec)'
            self.w_m = w_m
        
        if w_fb is not None:
            assert self.driving_feedback, 'driving_feedback should be set to True'
            assert w_fb.shape == (self.n_rec,self.n_out), 'Dimensions must be (n_rec,n_out)'
            self.w_fb = w_fb
    
    
    def next_state(self, x_in: np.array) -> None:
        """ 
        Advance the network forward by one step 
        
        Note that this is the basic RNN activity equation
        
        Args:
            x_in (np.array): external input
        """
        
        self.h_prev = np.copy(self.h)
        self.x_in_prev = self.x_in
        
        """ recurrent activity """                        
        if self.driving_feedback:
            # Feedback signal is position (which seems to do better than error)
            if self.feedback_signal == 'position':
                self.u = np.dot(self.w_rec, self.h) + np.dot(self.w_in, x_in + self.sig_in*self.rng.randn(self.n_in,1)) + np.dot(self.w_fb, self.pos + self.sig_fb*self.rng.randn(self.n_out,1)) 
            
            # Feedback signal here is error, not position
            if self.feedback_signal == 'error':
                self.u = np.dot(self.w_rec, self.h) + np.dot(self.w_in, x_in + self.sig_in*self.rng.randn(self.n_in,1)) + np.dot(self.w_fb, self.err + self.sig_fb*self.rng.randn(self.n_out,1)) 

        else:
            self.u = np.dot(self.w_rec, self.h) + np.dot(self.w_in, x_in + self.sig_in*self.rng.randn(self.n_in,1)) 

        # update step
        #self.xi = self.sig_rec*self.rng.randn(self.n_rec,1)
        self.xi = self._generate_recurrent_noise()
        
        self.h = self.h + (-self.h + self.f(self.u) + self.xi)*self.dt/self.tau_rec
        #self.h = self.h + (-self.h + self.f(self.u) + self.sig_rec*self.rng.randn(self.n_rec,1))/self.tau_rec
        
        self.x_in = x_in
       

    def output(self) -> None:
        
        """ Readout of the RNN
        
        If there is no velocity transform, the readout is just
        a mapping from the RNN activity directly to the position
        via the matrix 'w_out'
        """
        
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
            
            
            
            
            
            
    def _generate_recurrent_noise(self):

        """ Generate Recurrent Noise from multivariate gaussian

        This function generates noise that is injected into the recurrent units.
        Noise is sampled from a gaussian distribution, and can be nonisotropic or low-D.

        Returns:
        xi: vector of dimension n_rec, divided by square root of integration step dt
        
        Note: It is up to the user to check that a specified covariance matrix is positive semidefinite
        """

        # sample from multivariate gaussian
        mean = np.zeros(self.sig_rec_covariance.shape[0])
            
        xi = self.rng.multivariate_normal(mean, cov=self.sig_rec_covariance, size=1).T # should be size (n_neurons,1)
            
        return xi/np.sqrt(self.dt)
        


