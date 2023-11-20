# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:10:26 2020

@author: Amin
"""

import numpy as np

class Neuron:
    """Properties and methods related to a single neuron; 
    
        Properties:
            position (numpy.ndarray): 1x3 array of x,y,z coordinates of the neuron
            color (numpy.ndarray): 1xC array of the color intensity of neuron
            covariance (numpy.ndarray): 3x3 covariance represeting neuron's shape
            annotation (string): Expert annotation of the neuron name
            id (string): ID assigned to the neuron based on the computational methods
        
    """
    
   
    def __init__(self):
        """Construct an instance of this class
        """
                
         # Neuron position & color.
        self.position        = np.zeros((1,3))*np.nan # neuron pixel position (x,y,z)
        self.color           = np.zeros((1,4))*np.nan # neuron color based on fitting (R,G,B,W,...), W = white channel, values=[0-255]
        self.color_readout   = np.zeros((1,4))*np.nan # neuron color based on readout from the image
        self.baseline        = np.zeros((1,4))*np.nan # baseline noise values (R,G,B,W,...), values=[-1,1]
        self.covariance      = np.zeros((1,3,3))*np.nan # 3x3 covariance matrix that's fit to the neuron
        self.aligned         = np.zeros((7))*np.nan 
        
        # User neuron ID.
        self.annotation      = '' # neuron user selected annotation
        self.annotation_confidence = np.nan # user confidence about annotation
        
        # Auto neuron ID.
        self.deterministic_id    = None  # neuron ID assigned by the deterministic model
        self.probabilistic_ids   = None # neuron IDs listed by descending probability
        self.probabilistic_probs = None # neuron ID probabilities
    
    def annotate(obj, name, confidence):
        """Annotate the neuron
    
        Args:
            name: The neuron name
            confidence: The user confidence
            
        """
        
        # Remove the user annotation.
        if name is None or name == '':
            obj.annotation = ''
            obj.annotation_confidence = np.nan

        # Annotate the neuron.
        else:
            obj.annotation = name


    def delete_annotation(obj):
        """Delete the user ID
        """
        obj.annotate('', 0, np.nan);
    
    def delete_model_ID(obj):
        """Delete the model-predicted ID
        """

        obj.deterministic_id = None
        obj.probabilistic_ids = None
        obj.probabilistic_probs = None
        obj.rank = None
    