import glob
import h5py
from pynwb import NWBHDF5IO, validate, get_type_map
from hdmf.backends.hdf5.h5_utils import H5SpecWriter
from hdmf.backends.utils import NamespaceToBuilderHelper
import argparse
from typing import Union

import dandi 

import os

from pynwb import load_namespaces, get_class, register_class, NWBFile, TimeSeries, NWBHDF5IO
from pynwb.file import MultiContainerInterface, NWBContainer, Device, Subject
from pynwb.ophys import ImageSeries, OnePhotonSeries, OpticalChannel, ImageSegmentation, PlaneSegmentation, Fluorescence, CorrectedImageStack, MotionCorrection, RoiResponseSeries, ImagingPlane, DfOverF
from pynwb.core import NWBDataInterface
from pynwb.epoch import TimeIntervals
from pynwb.behavior import SpatialSeries, Position
from pynwb.image import ImageSeries

import shutil


def add_stimulus_SF(nwbfile, stim_time):

    heat_stim = TimeIntervals(
        name = 'heat_stimulus',
        description = 'Heat stimulation driven by 1436-nm 500-mW laser. Mean temperature set to be 10.0C over first second of stimulation with a 0.39 sec exponential decay, returning to baseline within 3 sec. Start and stop time are in seconds.'
    )

    heat_stim.add_row(start_time = stim_time, stop_time = stim_time+3)

    nwbfile.add_time_intervals(heat_stim)

    return nwbfile

if __name__ == '__main__':

    filepath = '/mnt/flavell/'
    savepath = '/mnt/flavell/'

    heat_stim_dict = {'2022-12-21-06': 802,
                      '2023-01-05-01': 801,
                      '2023-01-05-18': 801,
                      '2023-01-06-01': 801,
                      '2023-01-06-08': 801,
                      '2023-01-06-15': 802,
                      '2023-01-09-08': 803,
                      '2023-01-09-15': 801,
                      '2023-01-09-22': 801,
                      '2023-01-10-07': 801,
                      '2023-01-10-14': 802,
                      '2023-01-13-07': 801,
                      '2023-01-16-01': 801,
                      '2023-01-16-08': 802,
                      '2023-01-16-15': 801,
                      '2023-01-16-22': 801,
                      '2023-01-17-07': 801,
                      '2023-01-17-14': 801,
                      '2023-01-18-01': 801}

    for file in os.listdir(filepath):

        io = NWBHDF5IO(filepath+'/'+file, mode='r')
        nwbfile = io.read()

        identifier = nwbfile.identifier
        rate = nwbfile.acquisition['CalciumImageSeries'].rate

        if identifier in heat_stim_dict.keys():
            print(identifier)
            stim_time = heat_stim_dict[identifier]
            nwbfile = add_stimulus_SF(nwbfile, round(stim_time/rate))
            writeio = NWBHDF5IO(savepath + '/' + file, mode='w')
            writeio.write(nwbfile)
            writeio.close()

        else:
            shutil.copy(filepath + '/'+file, savepath+'/'+file)

        io.close()


    OpticalChannels, OpticalChannelRefs = build_channels(metadata)
    behavior, timestamps = build_behavior(data_path, file_name, metadata)

    heat_stim_dict = {'2022-12-21-06': 802,
                        '2023-01-05-01': 801,
                        '2023-01-05-18': 801,
                        '2023-01-06-01': 801,
                        '2023-01-06-08': 801,
                        '2023-01-06-15': 802,
                        '2023-01-09-08': 803,
                        '2023-01-09-15': 801,
                        '2023-01-09-22': 801,
                        '2023-01-10-07': 801,
                        '2023-01-10-14': 802,
                        '2023-01-13-07': 801,
                        '2023-01-16-01': 801,
                        '2023-01-16-08': 802,
                        '2023-01-16-15': 801,
                        '2023-01-16-22': 801,
                        '2023-01-17-07': 801,
                        '2023-01-17-14': 801,
                        '2023-01-18-01': 801}
                

    if file_name in heat_stim_dict.keys():
        stim_time = heat_stim_dict[identifier]
        nwb_file = add_stimulus_SF(nwb_file, round(stim_time/1.7))

    neuroPAL_module = nwb_file.create_processing_module(
        name = 'NeuroPAL',
        description = 'NeuroPAL image metadata and segmentation'
    )