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

import sys
sys.path.append("..")

from rnn import RNNparams, RNN
from task import Task
from simulation import Simulation
from algorithms.rflo import RFLO
from algorithms.reinforce import REINFORCE

from utils.plotting import plot_trained_trajectories

def test_init():
    params = RNNparams(n_in=2, n_rec=5,n_out=2,tau_rec=10,
                   eta_in=0.1,eta_rec=0.1,eta_out=0.1,
                   sig_in=0.0,sig_rec=0.0,sig_out=0.01,
                   rng=np.random.RandomState(14))
    
    assert params.n_in = 2
    assert params.sig_in == 0.0
    