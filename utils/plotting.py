import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FixedFormatter


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
sys.path.append("../..")


from task import Task

def plot_loss(loss,fig=None,title='Loss',label='Loss',ylabel='Loss',yscale='linear'):
    
    if not fig:
        fig = plt.figure(figsize=(6,3))
        
    plt.plot(loss,label=label)
    plt.xlabel('Trials')
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    plt.title(title)
    plt.legend()
    
    return fig

def interp_colors(dim,colormap='viridis'):
    
    """
    interp colors from viridis colormap
    """
    cmap = plt.get_cmap(colormap)
    colors = np.asarray(cmap.colors)
    
    x = np.arange(0, 256+1)
    f = interpolate.interp1d(x, x)
    xnew = np.linspace(0,dim,dim)*255/dim # so that lowest/highest values map to darkest/lightest colors
    ynew = f(xnew)
    ynew = np.asarray(list(map(int, ynew)))
    colors = colors[np.asarray(list(map(int, xnew))),:] # select every nth item 
    
    return colors


def insert_colorbar(fig,colormap='viridis',top_label='late',bottom_label='early'):
    '''
    Insert colorbar and label extremeties
    '''
    cmap = plt.get_cmap(colormap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    
    # fake up the array of the scalar mappable...
    # see https://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plotss
    sm._A = []
    
    cbar = fig.colorbar(sm)
    ll = cbar.ax.get_yticklabels()
    tick_labels = ['']*len(ll)
    tick_labels[0] = bottom_label
    tick_labels[-1] = top_label
    cbar.ax.set_yticklabels(tick_labels)
    
    return cbar




def plot_position(fig, pos, tasks: List[Task], count: Optional[int] = None, n_trials: Optional[int] = None, plot_freq: Optional[int] = None, **kwargs):
    
    
    
    """ Plot Trajectory (position) """
    
    ax = fig.gca()
        
    if count and n_trials and plot_freq:
        dim = int(n_trials/plot_freq) + 2
        
        colors = interp_colors(dim,colormap='viridis')        
        ax.plot(pos.squeeze()[:,0],pos.squeeze()[:,1],color = colors[int(count/plot_freq)],alpha=0.5)
        
        if int(count/plot_freq) == int(n_trials/plot_freq)-1:
            cbar = insert_colorbar(fig,colormap='viridis',**kwargs)
    
    else:
    
        ax.plot(pos.squeeze()[:,0],pos.squeeze()[:,1])
    
    #ax.set_title('RFLO, velocity={}, learning {}, {} trials'.format(net.velocity_transform,rflo.apply_to,i))
    
    for task in tasks:
        ax.scatter(task.y_target[0,:],task.y_target[1,:],s=100,marker='x',color='k')
    ax.scatter(0,0,s=100,marker='x',color='k')
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    
    ax.set_xlabel('position (x)')
    ax.set_ylabel('position (y)')
    ax.set_title('Trajectories During Training')

    return fig


def plot_trained_trajectories(sim, tasks: List[Task],colors=cycle(['teal','C4','darkblue','tomato','limegreen','magenta','aqua','maroon']),num_examples:int = 4, **kwargs):
    
    """ Plot trajectories after training """
    
    if 'fig' in kwargs.keys():
        fig = kwargs['fig']
        ax = fig.gca()
    else:
        fig,ax = plt.subplots(1,1,figsize=(5,5),squeeze=True) 
        
    if 'title' in kwargs.keys():
        title = kwargs['title']
    else:
        title = 'Trajectories after Training'
    
    for task in tasks:
        c = next(colors)
        ax.plot(task.y_target[0,:],task.y_target[1,:],'X',color=c,markersize=10,linewidth=5)
        
        for count in range(num_examples):

            sim.run_trial(task,probe_types=['h','pos'],train=False)
            ax.plot(sim.probes['pos'].squeeze()[:,0],sim.probes['pos'].squeeze()[:,1],'-',color=c)

    ax.set_xticks([-1,1])
    ax.set_yticks([-1,1])
    ax.plot([0],[0],'o',color='k',markersize=10,linewidth=1)
    ax.set_title(title)
    ax.axis('off')
    
    return fig

    
    
    
    
    
    
    
    
def paper_format(fig,ax,xlabels=None,ylabels=None,labelsize=10,ticksize=10,linewidth=2,xlim=None,ylim=[0,1],figsize=(2.5,2.5),tight_layout=True):
    
    """ Format Figure for Paper 8.5 x 11 
    
    This allows for quick reformatting of figures
    
    Args
    ----
    labelsize
    ticksize
    linewidth
    ylim
    figsize
    
    Returns
    -------
    fig
    ax
    
    TO DO: Need to be able to set linewidth
    """
    
    
    
    fig.set_figheight(figsize[0])
    fig.set_figwidth(figsize[1])
    
    ax.set_ylim(ylim)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    
    ax.xaxis.label.set_size(labelsize)
    ax.yaxis.label.set_size(labelsize)
    
    ax.tick_params(axis='x', labelsize=ticksize)
    ax.tick_params(axis='y', labelsize=ticksize)
    
    ax.set_title(ax.get_title(),fontsize=labelsize)
    
    """ for small figures """
    if xlabels:
        ax.xaxis.set_major_locator(FixedLocator(xlabels))
        ax.xaxis.set_major_formatter(FixedFormatter(xlabels))
    
    if ylabels:
        ax.yaxis.set_major_locator(FixedLocator(ylabels))
        ax.yaxis.set_major_formatter(FixedFormatter(ylabels))
    
    if ax.get_legend_handles_labels()[1] != []:
        ax.legend(prop={"size":labelsize})
    
    if tight_layout:
        fig.tight_layout()

    return fig,ax