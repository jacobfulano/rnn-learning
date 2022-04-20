import numpy as np

# miscellaneous
from itertools import cycle
from copy import deepcopy

import logging
import warnings
import dataclasses
from dataclasses import dataclass
from typing import Optional, List
from functools import reduce

from utils.metric import cos_sim, return_norm_and_angle

def theta(x):
    return 0.5*(1 + sign(x))


def f(x):
    return np.tanh(x)


def df(x):
    """The derivative of tanh(x) = 1/cosh(x)^2"""
    return 1/np.cosh(10*np.tanh(x/10))**2  # the tanh prevents overflow

def rgetattr(obj, attr):
    """A "recursive" version of getattr that can handle nested objects.
    Args:
        obj (object): Parent object
        attr (string): Address of desired attribute with '.' between child
            objects.
    Returns:
        The attribute of obj referred to."""

    return reduce(getattr, [obj] + attr.split('.'))



def realign_matrix(W,n_change,rng=np.random.RandomState(1964),zero=True,scale=1):
    
    """ Generate Partially Aligned Matrix
    
    This function generates a partially aligned matrix in one of two ways:
    
    method 1 - assign random elements of matrix to 0
    method 2 - assign random elements to new set of uniform random values, scaled by 'scale'
    
    
    Args
    ----
    W (np.array): matrix size 2 x neurons, representing the BMI decoder
    n_change (int): number of matrix values to change. The larger n_change, the less aligned the matrix
    rng: random number generator
    zero (bool): if True, applies method 1 of zeroing values in matrix
    scale (float): for method 2, scale new matrix
    
    Returns
    -------
    W_realigned (np.array): matrix of dimensions W

    """
    dim = W.shape[1]
    
    # Method 1
    if zero:
        ident = np.identity(dim)
        # remove rows
        idx = rng.choice(np.arange(0,dim),size=n_change,replace=False)
        ident[idx,:] = 0
        W_realigned = W @ ident
    
    # Method 2
    else:
        W_realigned = W.copy()
        idx = rng.choice(np.arange(0,len(W.ravel())),size=n_change,replace=False)
        W_realigned.ravel()[idx] = scale*(2*rng.rand(len(idx)) - 1)/dim**0.5 
        W_realigned=W_realigned.reshape(-1,dim)
    
    return W_realigned


def choose_aligned_matrix(W,
                   overlap=0.75,
                   n_change=30,
                   tolerance=0.05,
                   loop=1000,
                   rng=np.random.RandomState(1964),
                   zero=False,
                   verbose=True):
    
    """
    Find Partially Aligned Matrix
    
    This code uses the function `realign_matrix()` to find a matrix that overlaps with W by desired amount
    
    Args
    ----
    W (np.array): matrix size 2 x neurons, representing the BMI decoder
    overlap (float): scalar value between 0 and 1 representing alignment of W with new matrix M
    n_change (int): number or cells in matrix W to change. The more cells changed, the lower the alignment with W
    tolerance (float): how close realigned matrix should be to desired overlap
    loop (int): how many search iterations in for loop
    rng: random number generator
    zero (bool): if True, applies method 1 of zeroing values in matrix. Passed to `realign_matrix`
    verbose (bool): whether to print final alignment
    
    Returns
    -------
    M: partially aligned matrix with dimensions of W
    returns None if no matrix is found with desired overlap in specified iterations
    
    """
    
    assert overlap <= 1, 'overlap must be between 0 and 1'
    assert overlap >= 0, 'overlap must be between 0 and 1'

    if overlap == 1:
        n_change = 0
    
    for i in range(loop):

        M = realign_matrix(W,n_change=n_change,rng=rng,zero=zero)
        norm, angle = return_norm_and_angle(W,M)

        if angle < overlap + tolerance and angle > overlap - tolerance:
            if verbose: print('\rM norm: {:.2f}'.format(norm) + '\t M angle: {:.2f}, {} iterations'.format(angle,i),end='')
            return M
            
        else:
            if verbose: print('\rNO OVERLAP M angle: {:.2f}, {} iterations'.format(angle,i),end='')
                
    return None # if no successful matrix. This prevents code from running downstream
