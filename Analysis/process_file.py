import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pynwb import load_namespaces, get_class, register_class, NWBFile, TimeSeries, NWBHDF5IO



def get_nwb_neurons(filepath):
    '''
    Take in path to NWB file and return dataframe containing labels, xyz positions, and RGB values
    '''
    with NWBHDF5IO(filepath, mode='r', load_namespaces=True) as io:
        read_nwb = io.read()
        seg = read_nwb.processing['NeuroPAL']['NeuroPALSegmentation']['NeuroPALNeurons'].voxel_mask[:]
        labels = read_nwb.processing['NeuroPAL']['NeuroPALSegmentation']['NeuroPALNeurons']['ID_labels'][:]
        channels = read_nwb.acquisition['NeuroPALImageRaw'].RGBW_channels[:] #get which channels of the image correspond to which RGBW pseudocolors
        image = read_nwb.acquisition['NeuroPALImageRaw'].data[:]
        scale = read_nwb.imaging_planes['NeuroPALImVol'].grid_spacing[:] #get which channels of the image correspond to which RGBW pseudocolors
    
    labels = ["".join(label) for label in labels]

    blobs = pd.DataFrame.from_records(seg, columns = ['X', 'Y', 'Z', 'weight'])
    blobs = blobs.drop(['weight'], axis=1)

    RGB_channels = channels[:-1]

    RGB = image[:,:,:,RGB_channels]

    blobs[['R','G','B']] = [RGB[row['x'],row['y'],row['z'],:] for i, row in blobs.iterrows()]
    blobs[['xr', 'yr', 'zr']] = [[row['x']*scale[0],row['y']*scale[1], row['z']*scale[2]] for i, row in blobs.iterrows()]
    blobs['ID'] = labels

    blobs = blobs.replace('nan', np.nan, regex=True) 

    return blobs, RGB

def get_dataset_neurons(folder):
    dataset = {}
    for file in os.listdir(folder):
        if not file[-4:] =='.nwb':
            continue

        blobs, image = get_nwb_neurons(folder+'/'+file)

        dataset[file[:-4]] = blobs

    return dataset


def combine_datasets(datasets):
    for i, dataset in enumerate(datasets):
        if i ==0:
            upd_data = dataset
        else:
            upd_data.update(dataset)

    return upd_data

def get_pairings(dataset):
    pairings = {}

    for file, blobs in dataset.items():
        IDd = blobs[blobs['ID']!='']

        for i in range(len(IDd)):
            for j in range(i,len(IDd)):
                label1 = blobs.loc[i,'ID']
                label2 = blobs.loc[j, 'ID']

                xyz1 = np.asarray(blobs.loc[i,['xr','yr','zr']])
                xyz2 = np.asarray(blobs.loc[j,['xr','yr','zr']])
                
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