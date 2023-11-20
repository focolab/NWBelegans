# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:09:27 2020

@author: Amin
"""

from Atlas.Atlas import Atlas
from AUtils import Simulator

# %%

bodypart    = 'head'
n_samples   = 10

atlas       = Simulator.loat_atlas('Models/atlas_xx_rgb.mat',bodypart) # Load the pre-trained atlas
ims     = Simulator.simulate_gmm(atlas,n_samples=n_samples) # Simulate worms from the generative model

trained_atlas, aligned, params, cost, counts = Atlas.train_atlas(ims,bodypart) # Train atlas on the sample worms
trained_atlas, aligned, params = Atlas.major_axis_align(trained_atlas, aligned, params, shift=10)
Atlas.visualize_pretty(trained_atlas,aligned,'')



# %%
