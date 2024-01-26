import numpy as np
import pandas as pd
import scipy.io as sio
from utils import convert_coordinates
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pickle
from sklearn.decomposition import PCA


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
        self.df = self.get_df()

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
        df_sorted = df_conv_atlas.sort_values(by=['ganglion', 'h'])

        # Reset the index if needed
        df_sorted = df_sorted.reset_index(drop=True)

        self.df = df_sorted

        return self.df

class NWBAtlas:
    def __init__(self, atlas_file = '/Users/danielysprague/foco_lab/data/atlases/2023_11_02_1.pkl', ganglia = '/Users/danielysprague/foco_lab/data/atlases/neuron_ganglia.csv'):

        self.ganglia = pd.read_csv(ganglia)
        with open(atlas_file, 'rb') as f:
            loaded_dict = pickle.load(f)
        neurons = loaded_dict['names']
        mu = loaded_dict['mu']
        sigma = loaded_dict['sigma']

        atlas_color = mu[:,3:]
        atlas_color[atlas_color<0] = 0
        atlas_color = atlas_color/(np.percentile(atlas_color,95,axis=0)+1e-5)
        atlas_color[atlas_color>1] = 1
        self.atlas_color = atlas_color

        pca = PCA(n_components=3)
        pca.fit(mu[:,:3])
        projection = pca.components_.T

        self.projection = projection
        self.mu = mu
        self.sigma = sigma
        self.xyzmu = mu[:,:3].copy()@projection
        self.xyzmu[:,0] = self.xyzmu[:,0]-np.min(self.xyzmu[:,0])
        self.xyzmu[:,1] = self.xyzmu[:,1]-np.min(self.xyzmu[:,1])
        self.xyzmu[:,2] = -(self.xyzmu[:,2]-np.min(self.xyzmu[:,2]))
        self.rgbmu = mu[:,3:]
        cov = np.zeros((3,3,mu.shape[0]))
        for i in range(cov.shape[2]):
            cov[:,:,i] = projection.T@sigma[:3,:3,i].copy()@projection

        self.xyzsigma = cov
        self.rgbsigma = sigma[3:,3:,:]
        
        self.neurons = neurons
        self.neur_dict = {}
        self.df = self.get_df()
            
        #if plot_cov:
        #    for i in range(cov.shape[2]):
        #        Atlas.draw_ellipse(mus[i,:],cov[:,:,i],atlas_color[i,:3][None,:],
        #                           std_devs=1.5,ax=ax,line_width=2)

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

            index = np.argwhere(self.neurons == ID)

            self.neur_dict[ID] = {'xyz_mu': self.xyzmu[index,:], 'rgb_mu': self.rgbmu[index,:], 'xyz_sigma':self.xyzsigma[:,:,index], 'rgb_sigma': self.rgbsigma[:,:,index], 'class':neurClass, 'ganglion':gang}

        return self.neur_dict

    def get_df(self, vRecenter=[0,0,0]):
        
        df_gangl = pd.DataFrame(self.ganglia)
        mu = np.hstack((self.xyzmu,self.rgbmu))
        df_atlas = pd.DataFrame(mu, columns = ['X','Y','Z', 'R', 'G', 'B'])

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

        df_conv_atlas = convert_coordinates(df_atlas, vRecenter=vRecenter)

        df_atlas['ganglion'].fillna('other', inplace=True)

        custom_sort_order = ['Anterior Pharyngeal Bulb', 'Anterior Ganglion', 'Dorsal Ganglion', 'Lateral Ganglion', 'Ventral Ganglion', 'Retro Vesicular Ganglion', 'Posterior Pharyngeal Bulb', 'other']

        # Use the Categorical data type to specify the custom sort order
        df_conv_atlas['ganglion'] = pd.Categorical(df_conv_atlas['ganglion'], categories=custom_sort_order, ordered=True)

        # Sort the DataFrame first by 'class' and then by 'ID'
        df_sorted = df_conv_atlas.sort_values(by=['ganglion', 'h'])

        # Reset the index if needed
        df_sorted = df_sorted.reset_index(drop=True)

        self.df = df_sorted

        return self.df

    def draw_ellipse(self,mean,covariance,color,std_devs=1.5,ax=None,line_width=2):
        # sample grid that covers the range of points
        min_p = mean - std_devs*np.sqrt(np.diag(covariance))
        max_p = mean + std_devs*np.sqrt(np.diag(covariance))
        
        x = np.linspace(min_p[0],max_p[0],256) 
        y = np.linspace(min_p[1],max_p[1],256)
        X,Y = np.meshgrid(x,y)

        Z = multivariate_normal.pdf(np.stack((X.reshape(-1),Y.reshape(-1))).T, mean=mean, cov=(std_devs**2)*covariance)
        Z = Z.reshape([len(x),len(y)])

        if np.any(Z==0):
            Z[Z==0] = 0.0001

        if ax is None:
            plt.contour(X, Y, Z, 0,  colors=color,linewidth=line_width)
            #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            #ax.plot_surface(X, Y, Z,cmap='viridis',linewidth=0)
            #plt.show()
        else:
            ax.contour(X, Y, Z, 0, colors=color,linewidths=line_width)

    def project_atlas_components(self, xyz):

        newxyz = xyz.copy() @self.projection
        newxyz[:,0] = newxyz[:,0] -np.min(newxyz[:,0])
        newxyz[:,1] = newxyz[:,1] -np.min(newxyz[:,1])
        newxyz[:,2] = -(newxyz[:,2] -np.min(newxyz[:,2]))

        return newxyz
        