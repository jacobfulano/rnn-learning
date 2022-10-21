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


def cos_sim(a,b):
    """ cosine similarity between vectors """
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def return_norm_and_angle(A,B):
    """ Calculate two measures of similarity between two matrices"""
    
    norm = np.linalg.norm(A-B)
    
    AA = A.ravel()
    BB = B.ravel()
    angle = np.linalg.norm(AA.T@BB)/(np.linalg.norm(AA) * np.linalg.norm(BB))
    
    return norm, angle


def flow_field_data(h_list, verbose: bool = False, fit_intercept: bool = False):
    
    """ Autoregression
    
    This tries to predict h(t+1) from h(t) assuming a linear matrix h(t+1) = W h(t)
    
    Args
    ----
    h_list: a list of activity arrays h. Each item in the list is an h array for a single trial
    verbose: whether to print
    fit_intercept: bool, whether to fit intercept in linear regression
    
    Returns
    -------
    F (np.array): linear regression coefficients (i.e. matrix)
    score: linear regression score
    
    """
    n_steps = h_list[0].squeeze().shape[0] # size (n_steps,n_neurons) e.g.
    n_neurons = h_list[0].squeeze().shape[1]
    
    # Concatenate data across trials
    # note here that we are fitting 0:n-1 to 1:n
    
    X = h_list[0].squeeze()[0:n_steps-1,:]
    Y = h_list[0].squeeze()[1:n_steps,:]
    
    if verbose: print(activity.shape)

    for i in range(1,len(h_list)):
        X = np.vstack((X,h_list[i].squeeze()[0:n_steps-1,:]))
        Y = np.vstack((Y,h_list[i].squeeze()[1:n_steps,:]))
        
    # Apply linear regression without fitting the intercept, although this _could_
    lr = LinearRegression(fit_intercept=fit_intercept)
    lr.fit(X, Y)
    F = lr.coef_
    
    return F, lr.score(X,Y) # this score is not separate from training data




def flow_field_predicted(W,err_list,h_list):
    
    """ Prediction change in flow field due to learning
    
    Args
    ----
    W (np.array): matrix used to calculate flow field based on learning
    err_list: list of np.arrays, each item is err for a trial
    h_list: list of np.arrays, each item is h for a trial
    
    TO DO: Maybe consider alternative learning rule updates here?
    
    TO CHECK: Should this be divided by number of trials?
    """
    
    n_steps = h_list[0].squeeze().shape[0]
    n_neurons = h_list[0].squeeze().shape[1]
    
    dF = np.zeros((n_neurons,n_neurons))
    
    # loop through trials
    for trial in range(len(h_list)):
        
        # loop through timesteps
        for t in range(n_steps):
            
            # this assumes cumulative weight update
            dF += np.outer(W.T @ err_list[trial].squeeze()[t], h_list[trial].squeeze()[t])
            
            
    return dF
            
        

        
        
        
def calculate_flow_field_correlation(Fpred,Fdata,h_list):
    
    """ Calculate Correlation between two flow fields 
    
    Args
    ----
    Fpred (np.array): predicted flow field
    Fdata (np.array): calculate flow field
    h_list: 
    
    Returns
    -------
    mean across all correlation values (i.e. summed across trials and time)
    """
    
    n_steps = h_list[0].squeeze().shape[0]
    
    corr_list = []
    
    for trial in range(len(h_list)):
        
        for t in range(n_steps):
            
            A = Fpred @ h_list[trial].squeeze()[t]
            B = Fdata @ h_list[trial].squeeze()[t]
            
            corr_list.append(np.dot(A, B)/(np.linalg.norm(A) * np.linalg.norm(B)))
            
            
    return np.mean(corr_list)
            
    
