from collections.abc import Iterable
import os

from datetime import datetime, timedelta
from dateutil import tz
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.data_utils import DataChunkIterator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pynwb import load_namespaces, get_class, register_class, NWBFile, TimeSeries, NWBHDF5IO
from pynwb.file import MultiContainerInterface, NWBContainer, Device, Subject
from pynwb.ophys import ImageSeries, OnePhotonSeries, OpticalChannel, ImageSegmentation, PlaneSegmentation, Fluorescence, CorrectedImageStack, MotionCorrection, RoiResponseSeries, ImagingPlane, DfOverF
from pynwb.core import NWBDataInterface
from pynwb.epoch import TimeIntervals
from pynwb.behavior import SpatialSeries, Position
from pynwb.image import ImageSeries
import scipy.io as sio
import skimage.io as skio
from tifffile import TiffFile
import time

# ndx_mulitchannel_volume is the novel NWB extension for multichannel optophysiology in C. elegans
from ndx_multichannel_volume import CElegansSubject, OpticalChannelReferences, OpticalChannelPlus, ImagingVolume, VolumeSegmentation, MultiChannelVolume

def gen_file(description, identifier, start_date_time, lab, institution, pubs):

    nwbfile = NWBFile(
        session_description = description,
        identifier = identifier,
        session_start_time = start_date_time,
        lab = lab,
        institution = institution,
        related_publications = pubs
    )

    return nwbfile

def create_subject(nwbfile, description, identifier, date_of_birth, growth_stage, growth_stage_time, cultivation_temp, sex, strain):
    
    nwbfile.subject = CElegansSubject(
        subject_id = identifier,
        date_of_birth = date_of_birth,
        growth_stage = growth_stage,
        growth_stage_time = growth_stage_time,
        cultivation_temp = cultivation_temp,
        species = "http://purl.obolibrary.org/obo/NCBITaxon_6239",
        sex = sex,
        strain = strain
    )

    return 

def create_device(nwbfile, name, description, manufacturer):

    device = nwbfile.create_device(
        name = name,
        description = description,
        manufacturer = manufacturer
    )

    return device


def create_im_vol(nwbfile, name, device, channels, location="head", grid_spacing=[0.3208, 0.3208, 0.75], grid_spacing_unit ="micrometers", origin_coords=[0,0,0], origin_coords_unit="micrometers", reference_frame="Worm head"):
    
    # channels should be ordered list of tuples (name, description)

    OptChannels = []
    OptChanRefData = []
    for fluor, des, wave in channels:
        excite = float(wave.split('-')[0])
        emiss_mid = float(wave.split('-')[1])
        emiss_range = float(wave.split('-')[2][:-1])
        OptChan = OpticalChannelPlus(
            name = fluor,
            description = des,
            excitation_lambda = excite,
            excitation_range = [excite-1.5, excite+1.5],
            emission_range = [emiss_mid-emiss_range/2, emiss_mid+emiss_range/2],
            emission_lambda = emiss_mid
        )

        OptChannels.append(OptChan)
        OptChanRefData.append(wave)

    OptChanRefs = OpticalChannelReferences(
        name = 'OpticalChannelRefs',
        channels = OptChanRefData
    )

    ImagingVolume = ImagingVolume(
        name= name,
        optical_channel_plus = OptChannels,
        order_optical_channels = OptChanRefs,
        description = 'NeuroPAL image of C elegan brain',
        device = device,
        location = location,
        grid_spacing = grid_spacing,
        grid_spacing_unit = grid_spacing_unit,
        origin_coords = origin_coords,
        origin_coords_unit = origin_coords_unit,
        reference_frame = reference_frame
    )

    nwbfile.add_imaging_plane(ImagingVolume)

    return ImagingVolume, OptChanRefs

def create_image(name, description, data, ImagingVolume, OptChanRefs, resolution=[0.3208, 0.3208, 0.75], RGBW_channels=[0,1,2,3]):

    image = MultiChannelVolume(
        name = name,
        Order_optical_channels = OptChanRefs,
        resolution = resolution,
        description = description,
        RGBW_channels = RGBW_channels,
        data = data,
        imaging_volume = ImagingVolume
    )

    return image

def create_vol_seg_centers(name, description, ImagingVolume, positions, labels=None, reference_images = None):

    '''
    Use this function to create volume segmentation where each ROI is coordinates
    for a single neuron center in XYZ space.

    Positions should be a 2d array of size (N,3) where N is the number of neurons and
    3 refers to the XYZ coordinates of the neuron in that order.

    Labels should be an array of cellIDs in the same order as the neuron positions.
    '''

    vs = VolumeSegmentation(
        name = name,
        description = description,
        imaging_volume = ImagingVolume,
        labels = labels,
        reference_images = reference_images
    )

    for i in range(positions.shape[0]):
        voxel_mask = np.hstack((positions[i], 1))  # add weight of 1 to each ROI

        vs.add_roi(voxel_mask=voxel_mask)

    return vs

def create_calc_series(name, data, imaging_plane, unit, scan_line_rate, dimension, rate, resolution, compression = False):

    if compression:
        data = H5DataIO(data=data, compression="gzip", compression_opts=4)

    calcium_image_series = OnePhotonSeries(
        name = name,
        data = data,
        unit = unit,
        scan_line_rate = scan_line_rate,
        dimension = dimension,
        rate = rate,
        resolution= resolution,
        imaging_plane = imaging_plane
    )

    return calcium_image_series

def iter_calc_tiff(filename, numZ):

    #TiffFile object allows you to access metadata for the tif file and selectively load individual pages/series
    tif = TiffFile(filename)

    #In this dataset, one page is one XY plane and every 12 pages comprises one Z stack for an individual time point
    pages = len(tif.pages)
    timepoints = int(pages/numZ)

    pageshape = tif.pages[0].shape

    #We iterate through all of the timepoints and yield each timepoint back to the DataChunkIterator
    for i in range(timepoints):
        tpoint = np.zeros((pageshape[1],pageshape[0], numZ))
        for j in range(numZ):
            image = np.transpose(tif.pages[i*numZ+j].asarray())
            tpoint[:,:,j] = image

        #Make sure array ends up as the correct dtype coming out of this function (the dtype that your data was collected as)
        yield tpoint.astype('uint16')

    tif.close()

    return


def process_NP_FOCO_Ray(datapath, dataset, strain):

    identifier = dataset
    session_description = 'NeuroPAL and calcium imaging of immobilized worm with optogenetic stimulus'
    session_start_time = datetime(int(identifier[0:4]), int(identifier[4:6]), int(identifier[6:8]), int(identifier[9:11]), int(identifier[12:14]), int(identifier[15:]), tzinfo=tz.gettz("US/Pacific"))
    lab = 'FOCO lab'
    institution = 'UCSF'
    pubs = ''

    nwbfile = gen_file(session_description, identifier, session_start_time, lab, institution, pubs)

    subject_description = 'NeuroPAL worm in microfluidic chip'
    dob = datetime(int(identifier[0:4]), int(identifier[4:6]), int(identifier[6:8]), tzinfo=tz.gettz("US/Pacific"))
    growth_stage = 'YA'
    gs_time = pd.Timedelta(hours=2, minutes=30).isoformat()
    cultivation_temp = 20.
    sex = "O"
    strain = strain

    nwbfile = create_subject(nwbfile, subject_description, identifier, dob, growth_stage, gs_time, cultivation_temp, sex, strain)

    microname = "Spinning disk confocal"
    microdescrip = "Leica DMi8 Inverted Microscope with Yokogawa CSU-W1 SoRA, 40x WI objective 1.1 NA"
    manufacturer = "Leica, Yokogawa"

    microscope = create_device(nwbfile, microname, microdescrip, manufacturer)

    matfile = datapath + '/Manual_annotate/' +dataset +'/neuroPAL_image.mat'
    mat = sio.loadmat(matfile)
    scale = np.asarray(mat['info']['scale'][0][0]).flatten()

    if folder <'20230322':
        channels = [("mTagBFP2", "Chroma ET 460/50", "405-460-50m"), ("CyOFP1", "Chroma ET 605/70","488-605-70m"), ("GFP-GCaMP", "Chroma ET 525/50","488-525-50m"), ("mNeptune 2.5", "Chroma ET 700/75", "561-700-75m"), ("Tag RFP-T", "Chroma ET 605/70", "561-605-70m"), ("mNeptune 2.5-far red", "Chroma ET 700/75", "639-700-75m")]
        RGBW_channels = [0,1,3,4]
    else:
        channels = [("mTagBFP2", "Chroma ET 460/50", "405-460-50m"), ("CyOFP1", "Chroma ET 605/70","488-605-70m"), ("CyOFP1-high filter", "Chroma ET 700/75","488-700-75m"), ("GFP-GCaMP", "Chroma ET 525/50","488-525-50m"), ("mNeptune 2.5", "Chroma ET 700/75", "561-700-75m"), ("Tag RFP-T", "Chroma ET 605/70", "561-605-70m"), ("mNeptune 2.5-far red", "Chroma ET 700/75", "639-700-75m")]
        RGBW_channels = [0,1,4,6]

    NP_ImVol, NP_OptChanRef = create_im_vol(nwbfile, 'NPImVol', microscope, channels, location="head", grid_spacing = scale)

    raw_file = datapath + '/NP_Ray/' + dataset + '/full_comp.tif'
    data = skio.imread(raw_file)
    data = np.tranpose(data)

    ImDescrip = 'NeuroPAL structural image'

    NP_image = create_image('NeuroPALImageRaw', ImDescrip, data, NP_ImVol, NP_OptChanRef, resolution=scale, RGBW_channels=RGBW_channels)

    nwbfile.add_acquisition(NP_image)

    blob_file = datapath + '/Manual_annotate/' + dataset + '/blobs.csv'
    blobs = pd.read_csv(blob_file)

    IDs = np.asarray(blobs['ID'])
    positions = np.asarray(blobs[['X', 'Y', 'Z']])

    vs_descrip = 'Neuron centers for multichannel volumetric image. Weight set at 1 for all voxels. Labels refers to cell ID of segmented neurons.'

    NeuroPALImSeg = ImageSegmentation(
        name = 'NeuroPALSegmentation',
        plane_segmentations = create_vol_seg_centers('NeuroPAL_neurons', vs_descrip, NP_ImVol, positions, IDs)
    )

    neuroPAL_module = nwbfile.create_processing_module(
        name = 'NeuroPAL',
        description = 'NeuroPAL image data and metadata'
    )

    neuroPAL_module.add(NeuroPALImSeg)
    neuroPAL_module.add(NP_OptChanRef)

    Proc_ImVol, Proc_OptChanRef = create_im_vol(nwbfile, 'ProcImVol', microscope, [channels[i] for i in RGBW_channels])

    proc_file = datapath+ '/NP_Ray/' + dataset + '/neuroPAL_image.tif'
    proc_data = np.tranpose(skio.imread(proc_file), [2,1,0,3])

    ProcDescrip = 'NeuroPAL image with median filtering followed by color histogram matching to reference NeuroPAL images'

    Proc_image = create_image('ProcessedImage', ProcDescrip, proc_data, Proc_ImVol, Proc_OptChanRef, resolution=scale, RGBW_channels=[0,1,2,3])

    processed_im_module = nwbfile.create_processing_module(
        name = 'ProcessedImage',
        description = 'Data and metadata associated with the pre-processed neuroPAL image.'
    )

    processed_im_module.add(Proc_image)
    processed_im_module.add(Proc_OptChanRef)

    GCaMP_chan = [("GFP-GCaMP", "Chroma ET 525/50","488-525-50m")]

    Calc_scale = scale #DOUBLE CHECK ON THIS

    Calc_descrip = 'Imaging volume used to acquire calcium imaging data'

    Calc_ImVol, Calc_OptChanRef = create_im_vol(nwbfile, 'CalciumImVol', Calc_descrip, GCaMP_chan, grid_spacing=Calc_scale)

    Calc_file = datapath + '/NP_Ray/' + dataset + '.tiff'

    tif = TiffFile(Calc_file)

    page = tif.pages[0]
    numx = page.shape()[0]
    numy = page.shape()[1]
    numz = len(tif.pages)/1500

    data = DataChunkIterator(
        data = iter_calc_tiff(Calc_file, numz),
        maxshape = None,
        buffer_size = 10
    )

    Calc_name = 'CalciumImageSeries'

    Calc_ImSeries = create_calc_series(Calc_name, data, Calc_ImVol, "n/a", 0.5, [numx, numy, numz], 1.04, 1, compression=True)

    nwbfile.add_acquisition(Calc_ImSeries)

    gce_file = datapath + '/NP_Ray/' + dataset +'/extractor-objects/' + dataset + '_gce_quantification.csv'

    gce_quant = pd.read_csv(gce_file)

    gce_df = gce_quant[['X', 'Y', 'Z', 'gce_quant', 'ID', 'T', 'blob_ix']]

    blobquant = None
    for idx in gce_quant['blob_ix'].unique():
        blob = gce_df[gce_df['blob_ix']==idx]
        blobarr = np.asarray(blob[['X','Y','Z','gce_quant','ID']]) 
        blobarr = blobarr[np.newaxis, :, :]
        if blobquant is None:
            blobquant=blobarr

        else:
            blobquant = np.vstack((blobquant, blobarr))

    volsegs = []

    for t in range(blobquant.shape[1]):
        blobs = np.squeeze(blobquant[:,t,0:3])

        vsname = 'Seg_tpoint_'+str(t)
        description = 'Neuron segmentation for time point ' +str(t) + ' in calcium image series'
        volseg = create_vol_seg_centers(vsname, description, Calc_ImVol, blobs, reference_images=Calc_ImSeries)

        volsegs.append(volseg)

    CalcImSeg = ImageSegmentation(
        name = 'CalciumSeries_neurons',
        plane_segmentations = volsegs
    )

    gce_data = np.tranpose(blobquant[:,:,3])

    rt_region = volsegs[0].create_roi_table_region(
        description = 'Segmented neurons associated with calcium image series. This rt_region uses the location of the neurons at the first time point',
        region = list(np.arange(blobquant.shape[0]))
    )

    RoiResponse = RoiResponseSeries( # CHANGE WITH FEEDBACK FROM RAY
        name = 'CalciumImResponseSeries',
        description = 'Fluorescence activity for calcium imaging data',
        data = gce_data,
        rois = rt_region,
        unit = 'Percentage',
        rate = 1.04
    )

    fluor = Fluorescence(
        name = 'CalciumFluorTimeSeries',
        roi_response_series = RoiResponse
    )

    calcium_im_module = nwbfile.create_processing_module(
    name = 'CalciumActivity',
    description = 'Data and metadata associated with time series of calcium images'
)

    calcium_im_module.add(CalcImSeg)
    calcium_im_module.add(fluor)
    calcium_im_module.add(Calc_OptChanRef)

    io = NWBHDF5IO(datapath + '/NWB_Ray/'+identifier+'.nwb', mode='w')
    io.write(nwbfile)
    io.close()



if __name__ == '__main__':
    datapath = '/Users/danielysprague/foco_lab/data'

    for folder in os.listdir(datapath+'/NP_Ray'):
        if folder == '.DS_Store':
            continue

        print(folder)
        t0 = time.time()
        process_NP_FOCO_Ray(datapath, folder, strain)
        t1 = time.time()
        print(t1-t0)
