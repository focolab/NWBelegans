import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pynwb import NWBHDF5IO

class test_NWB():

    def __init__(self, nwbfile):

        with NWBHDF5IO(nwbfile, mode='r', load_namespaces=True) as io:
            read_nwbfile = io.read()

            subject = read_nwbfile.subject #get the metadata about the experiment subject
            growth_stage = subject.growth_stage
            image = read_nwbfile.acquisition['NeuroPALImageRaw'].data[:] #get the neuroPAL image as a np array
            channels = read_nwbfile.acquisition['NeuroPALImageRaw'].RGBW_channels[:] #get which channels of the image correspond to which RGBW pseudocolors
            im_vol = read_nwbfile.acquisition['NeuroPALImageRaw'].imaging_volume #get the metadata associated with the imaging acquisition
            
            seg = read_nwbfile.processing['NeuroPAL']['NeuroPALSegmentation']['NeuroPALNeurons'].voxel_mask[:] #get the locations of neuron centers
            labels = read_nwbfile.processing['NeuroPAL']['NeuroPALSegmentation']['NeuroPALNeurons']['ID_labels'][:]
            optchans = im_vol.optical_channel_plus[:] #get information about all of the optical channels used in acquisition
            chan_refs = read_nwbfile.processing['NeuroPAL']['OpticalChannelRefs'].channels[:] #get the order of the optical channels in the image
            calcium_frames = read_nwbfile.acquisition['CalciumImageSeries'].data[0:15, :,:,:] #load the first 15 frames of the calcium images
            print(read_nwbfile.acquisition['CalciumImageSeries'].dimension[:])
            fluor = read_nwbfile.processing['CalciumActivity']['Fluorescence']['GCaMP_activity'].data[:]
            #calc_seg = read_nwbfile.processing['CalciumActivity']['CalciumSeriesSegmentation']['Seg_tpoint_0'].voxel_mask[:]