# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:13:09 2020

@author: Amin
"""
from Neurons import Neuron, Image
import pyro.distributions as dist
from scipy.io import loadmat
import torch

# %%
def loat_atlas(file,bodypart):
    """Load C. elegans atlas file
        
    Args:
        file (string): File address of the file to be loaded
        bodypart (string): The bodypart that the atlas is requested for, choose
            between ('head','tail')
    
    Returns:
        dict: Atlas information with the following keys
            mu (numpy.ndarray): Nx(3+C) where N is the number of neurons and C 
                is the number of channles; each row corresponds to the center 
                and color of one neuron in the canonical space
            sigma (numpy.ndarray): Nx(3+C)x(3+C) where N is the number of neurons 
                and C is the number of channles; each row corresponds to the covariance
                of the position and color of one neuron in the canonical space
            names (array): String array containing the names of the neurons
            bodypart (string): Same as the input bodypart
    """
    
    content = loadmat(file,simplify_cells=True)
    
    mu = content['atlas'][bodypart]['model']['mu']
    sigma = content['atlas'][bodypart]['model']['sigma']
    names = content['atlas'][bodypart]['N']
    
    mu[:,:3] = mu[:,:3] - 1 # Matlab to Python
    
    return {'mu':mu, 'sigma':sigma, 'names': names, 'bodypart':bodypart}

# %%
def simulate_gmm(atlas,n_samples=10):
    """Simulate samples from atlas by sampling from Gaussian distributions and
        transforming according to random rotations
        
    Args:
        atlas (dict): Pre-trained statistical atlas
        n_samples (integer): Number of samples (worms) to be generated
        
    Returns:
        samples (numpy.ndarray): Positions and colors of the sampled worms
            with size (N,3+C,K)
    """
    
    # Sampling data
    
    C       = atlas['mu'].shape[1]-3 # Number of colors
    K       = atlas['mu'].shape[0] # Number of components

    μ_p = torch.tensor(atlas['mu']).float()
    Σ_p = torch.tensor(atlas['sigma']).float()
    
    mu = torch.zeros(K,C+3,n_samples)
    for k in range(K):
        mu[k,:,:] = dist.MultivariateNormal(µ_p[k,:], Σ_p[:,:,k]).sample((1,n_samples)).T.squeeze()
    
    
    #  Creating Images
    ims = []
    for n in range(n_samples):
        neurons = []
        for k in range(len(atlas['names'])):
            neuron = Neuron.Neuron()
             # Neuron position & color
            neuron.position        = mu[k,:3,n].numpy()
            neuron.color           = mu[k,3:,n].numpy()
            neuron.color_readout   = mu[k,3:,n].numpy()
            
            # User neuron ID
            neuron.annotation      = atlas['names'][k] 
            neuron.annotation_confidence = .99
            
            neurons.append(neuron)
            
        im = Image.Image(atlas['bodypart'],neurons)
        ims.append(im)
        
    return ims

