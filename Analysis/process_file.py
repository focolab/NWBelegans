import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pynwb import load_namespaces, get_class, register_class, NWBFile, TimeSeries, NWBHDF5IO
from sklearn.neighbors import NearestNeighbors
from dandi.dandiapi import DandiAPIClient
import remfile
import h5py


def get_nwb_neurons(filepath, atlas_neurons):
    '''
    Take in path to NWB file and return dataframe containing labels, xyz positions, and RGB values
    '''
    with NWBHDF5IO(filepath, mode='r', load_namespaces=True) as io:
        read_nwb = io.read()
        identifier = read_nwb.identifier
        seg = read_nwb.processing['NeuroPAL']['NeuroPALSegmentation']['NeuroPALNeurons'].voxel_mask[:]
        labels = read_nwb.processing['NeuroPAL']['NeuroPALSegmentation']['NeuroPALNeurons']['ID_labels'][:]
        channels = read_nwb.acquisition['NeuroPALImageRaw'].RGBW_channels[:] #get which channels of the image correspond to which RGBW pseudocolors
        image = read_nwb.acquisition['NeuroPALImageRaw'].data[:]
        scale = read_nwb.imaging_planes['NeuroPALImVol'].grid_spacing[:] #get which channels of the image correspond to which RGBW pseudocolors
    
    print(identifier)
    labels = ["".join(label) for label in labels]

    blobs = pd.DataFrame.from_records(seg, columns = ['X', 'Y', 'Z', 'weight'])
    blobs = blobs.drop(['weight'], axis=1)

    RGB_channels = channels[:-1]

    RGB = image[:,:,:,RGB_channels]

    blobs = blobs[(blobs['x']<RGB.shape[0])&(blobs['y']<RGB.shape[1])&(blobs['z']<RGB.shape[2])]

    idx_keep = [i for i, row in blobs.iterrows() if (row['x']<RGB.shape[0]) and (row['y']<RGB.shape[1]) and (row['z']<RGB.shape[2])]

    blobs[['R','G','B']] = [RGB[row['x'],row['y'],row['z'],:] for i, row in blobs.iterrows()]
    blobs[['xr', 'yr', 'zr']] = [[row['x']*scale[0],row['y']*scale[1], row['z']*scale[2]] for i, row in blobs.iterrows()]
    blobs['ID'] = [labels[i] for i in idx_keep]

    blobs = blobs.replace('nan', '', regex=True) 

    #blobs = blobs[blobs['ID'].isin(atlas_neurons)]

    return blobs, RGB

def get_dataset_neurons(folder, atlas_neurons):
    dataset = {}
    for file in os.listdir(folder):
        if not file[-4:] =='.nwb':
            continue

        blobs, image = get_nwb_neurons(folder+'/'+file, atlas_neurons)

        dataset[file[:-4]] = blobs

    return dataset

def get_dataset_online(dandi_id, atlas_neurons):
    dataset = {} 
    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(dandi_id, 'draft')
        for asset in dandiset.get_assets():
            s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
            file = remfile.File(s3_url)

            with h5py.File(file, 'r') as f:
                with NWBHDF5IO(file=f, mode='r', load_namespaces=True) as io:

                    read_nwb = io.read()
                    identifier = read_nwb.identifier
                    seg = read_nwb.processing['NeuroPAL']['NeuroPALSegmentation']['NeuroPALNeurons'].voxel_mask[:]
                    labels = read_nwb.processing['NeuroPAL']['NeuroPALSegmentation']['NeuroPALNeurons']['ID_labels'][:]
                    channels = read_nwb.acquisition['NeuroPALImageRaw'].RGBW_channels[:] #get which channels of the image correspond to which RGBW pseudocolors
                    image = read_nwb.acquisition['NeuroPALImageRaw'].data[:]
                    scale = read_nwb.imaging_planes['NeuroPALImVol'].grid_spacing[:] #get which channels of the image correspond to which RGBW pseudocolors

                print(identifier)
                
                labels = ["".join(label) for label in labels]

                labels = [label[:-1] if label.endswith('?') else label for label in labels]

                blobs = pd.DataFrame.from_records(seg, columns = ['X', 'Y', 'Z', 'weight'])
                blobs = blobs.drop(['weight'], axis=1)

                RGB_channels = channels[:-1]
                RGB = image[:,:,:,RGB_channels]

                blobs = blobs[(blobs['x']<RGB.shape[0])&(blobs['y']<RGB.shape[1])&(blobs['z']<RGB.shape[2])]

                idx_keep = [i for i, row in blobs.iterrows() if (row['x']<RGB.shape[0]) and (row['y']<RGB.shape[1]) and (row['z']<RGB.shape[2])]

                blobs[['R','G','B']] = [RGB[row['x'],row['y'],row['z'],:] for i, row in blobs.iterrows()]
                blobs[['xr', 'yr', 'zr']] = [[row['x']*scale[0],row['y']*scale[1], row['z']*scale[2]] for i, row in blobs.iterrows()]
                blobs['ID'] = [labels[i] for i in idx_keep]

                blobs = blobs.replace('nan', '', regex=True) 

                #blobs = blobs[blobs['ID'].isin(atlas_neurons)]

                dataset[identifier] = blobs

    return dataset

def combine_datasets(datasets):

    for i, dataset in enumerate(datasets):
        if i ==0:
            upd_data = dataset.copy()
        else:
            upd_data.update(dataset)

    return upd_data

def get_neur_nums(tot_dataset, atlas):

    neur_IDs = atlas.df['ID']

    num_datasets = len(tot_dataset.keys())
    neurons = {k:0 for k in neur_IDs}

    for dataset in tot_dataset.keys():
        blobs = tot_dataset[dataset]

        for i, row in blobs.iterrows():
            ID = row['ID']
            if ID == '':
                continue

            if not ID in neurons:
                neurons[ID] = 1

            else:
                neurons[ID] += 1

    return neurons, num_datasets

def get_pairings(dataset):
    pairings = {}

    for file, blobs in dataset.items():
        IDd = blobs[blobs['ID']!='']

        IDd.reset_index()

        for i in range(len(IDd)):
            for j in range(i,len(IDd)):

                label1 = IDd.loc[IDd.index[i],'ID']
                label2 = IDd.loc[IDd.index[j], 'ID']

                xyz1 = np.asarray(IDd.loc[IDd.index[i],['xr','yr','zr']])
                xyz2 = np.asarray(IDd.loc[IDd.index[j],['xr','yr','zr']])
                
                dist = np.linalg.norm(xyz1 - xyz2)

                pair1 = label1 + '-' +label2
                pair2 = label2 + '-' + label1

                if not (pair1 in pairings) or not (pair2 in pairings):
                    if i==j:
                        pairings[pair1] = [dist]
                    else:
                        pairings[pair1] = [dist]
                        pairings[pair2] = [dist]

                else:
                    if i==j:
                        pairings[pair1].append(dist)
                    else:
                        pairings[pair1].append(dist)
                        pairings[pair2].append(dist)

    return pairings

def get_color_discrim(folder, numneighbors): 

    color_discrim = {}

    for file in os.listdir(folder):
        if not file[-4:] == '.nwb':
            continue

        print(file)

        blobs, rgb_data = get_nwb_neurons(folder+'/'+file)

        color_norm = (rgb_data - np.min(rgb_data, axis=(0,1,2))) / (np.max(rgb_data, axis=(0,1,2))- np.min(rgb_data, axis=(0,1,2)))

        blobs[['Rnorm', 'Gnorm','Bnorm']] = np.nan

        color_data = np.zeros(len(blobs))

        for i, row in blobs.iterrows():
            colors = color_norm[max(row['x']-2,0):min(row['x']+2,rgb_data.shape[0]-1),max(row['y']-2,0):min(row['y']+2,rgb_data.shape[1]-1),max(row['z']-1,0):min(row['z']+1,rgb_data.shape[2]-1),:]

            flat_colors = colors.reshape(-1, colors.shape[-1])
            
            Rnorm = np.median(flat_colors[0])
            Gnorm = np.median(flat_colors[1])
            Bnorm = np.median(flat_colors[2])

            blobs.loc[i, 'Rnorm'] = Rnorm
            blobs.loc[i, 'Gnorm'] = Gnorm
            blobs.loc[i, 'Bnorm'] = Bnorm

        neighbors = NearestNeighbors(n_neighbors=numneighbors, algorithm='auto')

        X = np.asarray(blobs[['xr', 'yr', 'zr']])
        neighbors.fit(X)

        neighbor_dists, neighbor_index = neighbors.kneighbors(X=X, return_distance=True) #n_query x n_neighbors size matrix of distances to neighbors and indices in training data of closest neighbors

        for i, row in blobs.iterrows():
            neighbors= neighbor_index[i,1:]
            neighb_dists = neighbor_dists[i, 1:]
            neighb_dist_norm = 1- neighb_dists/sum(neighb_dists) #normalize distances of neighbors and then flip so that closest neighbors have highest weight
            color_dists = np.zeros(numneighbors-1)
            rgb = np.asarray(row[['Rnorm', 'Gnorm', 'Bnorm']])
            for j, neighbor in enumerate(neighbors):
                n_rgb = np.asarray(blobs.loc[neighbor,['Rnorm','Gnorm','Bnorm']])
                color_dists[j] = np.linalg.norm(rgb-n_rgb)
            avg_dist = np.dot(neighb_dist_norm, color_dists) #weighted average color distances to k nearest neighbors - proxy for discriminability to closest neighbors based on color

            color_data[i] = avg_dist

        color_discrim[file] = color_data

    return color_discrim                
