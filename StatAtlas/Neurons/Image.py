# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:10:51 2020

@author: Amin
"""

import numpy as np

class Image:
    """A list of neurons in a certain body part
    
        Properties:
            scale: image scale (x,y,z)
            neurons (Neuron): Array of neurons in the image (see Neurons.Neuron)
        
    """

    def __init__(self,bodypart,neurons=[],scale=np.ones((3))):
        """Construct an instance of this class.
    
        Args:
            bodypart: A string that represents which body part the
                current instance of this class corresponds to, examples are
                'head' and 'tail'
            scale: image scale (x,y,z)
            
        """

        # Initialize the data.
        self.bodypart = bodypart # a string consisting the name of the worm's body part
        
        # Set the scale.
        self.scale = scale
        
        # Set the neurons
        self.neurons = neurons
        
    
    def get_positions(obj,scale=1):
        """Getter of neuron positions
        """
        return np.array([neuron.position*scale for neuron in obj.neurons])
    
    def get_colors(obj):
        """Getter of neuron positions
            
            NOTE: THIS FUNCTION MAY RETURN NANS!!!
        """
        
        return np.array([neuron.color for neuron in obj.neurons])
    
    def get_colors_readout(obj):
        """Getter of neuron color readouts
        """
        return np.array([neuron.color_readout for neuron in obj.neurons])
    
    def get_covariances(obj):
        """Getter of neuron covariances
        
        NOTE: THIS FUNCTION MAY RETURN NANS!!!
        """
        return np.array([neuron.covariance for neuron in obj.neurons])
    
    def get_aligned(obj):
        """Getter for aligned neurons
        """
        return np.array([neuron.aligned for neuron in obj.neurons])
    
    def get_annotations(obj):
        """Getter of neuron annotations
        
        NOTE: THIS FUNCTION MAY RETURN < NUM(NEURONS)!!!
        """
        
        return [neuron.annotation for neuron in obj.neurons]

    def get_annotation_confidences(obj):
        """Getter of neuron annotation_confidences
        """
        
        return np.array([neuron.annotation_confidence for neuron in obj.neurons])
    