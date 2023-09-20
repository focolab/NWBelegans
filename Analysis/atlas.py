import numpy as np
import pandas as pd
import scipy.io as sio
from utils import convert_coordinates


'''
Helper functions for reading .mat function taken from 
https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
'''

def _check_keys( dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def loadmat(filename):
    """
    this function should be called instead of direct scipy.io .loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def get_NP_atlas(atlas_file):

    atlasfile = loadmat(atlas_file)
    atlas = atlasfile['atlas']
    head = atlas['head']
    neurons = head['N']
    model = head['model']
    mu = model['mu']
    sigma = model['sigma']
    
    return neurons, mu, np.transpose(sigma)

class NPAtlas:

    def __init__(self, atlas_file = 'data/atlases/atlas_xx_rgb.mat', ganglia = 'data/atlases/neuron_ganglia.csv'):
        
        self.ganglia = pd.read_csv(ganglia)
        atlasfile = loadmat(atlas_file)
        atlas = atlasfile['atlas']
        head = atlas['head']
        neurons = head['N']
        model = head['model']
        mu = model['mu']
        sigma = model['sigma']

        self.mu = np.asarray(mu)
        self.sigma = np.asarray(sigma)
        self.neurons = neurons
        self.neur_dict = {}
        self.df = None

    def create_dictionary(self):
        '''
        Create dictionary where keys are neuron IDs and values are dictionary of 
        neuron atlas attributes (xyz_mu, rgb_mu, xyz_sig, rgb_sig, class, ganglia)
        useful for 
        '''
        gang_dict = dict(zip(self.ganglia['neuron_class'].values, self.ganglia['ganglion'].values))
        
        for i, neuron in enumerate(self.neurons):

            ID = neuron

            if ID[-1] in ['L', 'R'] and ID[:-1]+'L' in self.neurons and ID[:-1]+'R' in self.neurons:        
                neurClass = ID[:-1]
                
            else:
                neurClass = ID

            gang = gang_dict.get(neurClass, 'Other')

            xyz_mu = self.mu[i,0:3]
            rgb_mu = self.mu[i,3:6]

            xyz_sigma = self.sigma[0:3, 0:3, i]
            rgb_sigma = self.sigma[3:6, 3:6, i]

            self.neur_dict[ID] = {'xyz_mu': xyz_mu, 'rgb_mu': rgb_mu, 'xyz_sigma':xyz_sigma, 'rgb_sigma': rgb_sigma, 'class':neurClass, 'ganglion':gang}

        return self.neur_dict

    def get_df(self):
        
        df_gangl = pd.DataFrame(self.ganglia)
        df_atlas = pd.DataFrame(self.mu, columns = ['X','Y','Z', 'R', 'G', 'B'])

        # find the LR paired neurons and assign neuron_class
        all_neurons = self.neurons
        neuron_class, is_LR, is_L, is_R = [], [], [], []
        for i in range(len(self.neurons)):
            ID = self.neurons[i]
            if ID[-1] in ['L', 'R'] and ID[:-1]+'L' in all_neurons and ID[:-1]+'R' in all_neurons:        
                neuron_class.append(ID[:-1])
                is_LR.append(1)
                if ID[-1] == 'L':
                    is_L.append(1)
                    is_R.append(0)
                if ID[-1] == 'R':
                    is_R.append(1)
                    is_L.append(0)
            else:
                neuron_class.append(ID)
                is_LR.append(0)
                is_L.append(0)
                is_R.append(0)

        df_atlas['neuron_class'] = neuron_class
        df_atlas['is_LR'] = is_LR
        df_atlas['is_L'] = is_L
        df_atlas['is_R'] = is_R
        df_atlas['ID'] = self.neurons

        # add ganglion column
        gang_dict = dict(zip(df_gangl['neuron_class'].values, df_gangl['ganglion'].values))
        df_atlas['ganglion'] = [gang_dict.get(k, 'other') for k in df_atlas['neuron_class']]  

        df_conv_atlas = convert_coordinates(df_atlas)

        df_atlas['ganglion'].fillna('other', inplace=True)

        custom_sort_order = ['Anterior Pharyngeal Bulb', 'Anterior Ganglion', 'Dorsal Ganglion', 'Lateral Ganglion', 'Ventral Ganglion', 'Retro Vesicular Ganglion', 'Posterior Pharyngeal Bulb', 'other']

        # Use the Categorical data type to specify the custom sort order
        df_conv_atlas['ganglion'] = pd.Categorical(df_conv_atlas['ganglion'], categories=custom_sort_order, ordered=True)

        # Sort the DataFrame first by 'class' and then by 'ID'
        df_sorted = df_conv_atlas.sort_values(by=['ganglion', 'ID'])

        # Reset the index if needed
        df_sorted = df_sorted.reset_index(drop=True)

        self.df = df_sorted

        return self.df
