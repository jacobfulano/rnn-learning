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

from task import Task


def plot_position(fig, pos, tasks: List[Task]):
    
    """ Plot Trajectory """
    
    ax = fig.gca()
    
    #self.probes['pos'].squeeze()[:,0],self.probes['pos'].squeeze()[:,1],
    
    ax.plot(pos.squeeze()[:,0],pos.squeeze()[:,1])
    #ax.set_title('RFLO, velocity={}, learning {}, {} trials'.format(net.velocity_transform,rflo.apply_to,i))
    for task in tasks:
        ax.scatter(task.y_target[0,:],task.y_target[1,:],s=100,marker='x',color='k')
    ax.scatter(0,0,s=100,marker='x',color='k')
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    #ax.legend()
    #plt.show()
    
    return fig