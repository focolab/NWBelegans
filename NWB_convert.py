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
import tifffile
import time

# ndx_mulitchannel_volume is the novel NWB extension for multichannel optophysiology in C. elegans
from ndx_multichannel_volume import CElegansSubject, OpticalChannelReferences, OpticalChannelPlus, ImagingVolume, VolumeSegmentation, MultiChannelVolume, MultiChannelVolumeSeries, SegmentationLabels

def gen_file(description, experimenter, exp_descript, identifier, start_date_time, lab, institution, pubs):

    nwbfile = NWBFile(
        session_description = description,
        identifier = identifier,
        session_start_time = start_date_time,
        experimenter = experimenter,
        experiment_description = exp_descript,
        lab = lab,
        institution = institution,
        related_publications = pubs
    )

    return nwbfile

def create_subject(nwbfile, description, identifier, date_of_birth, growth_stage, growth_stage_time, cultivation_temp, sex, strain):
    
    nwbfile.subject = CElegansSubject(
        subject_id = identifier,
        description = description,
        date_of_birth = date_of_birth,
        growth_stage = growth_stage,
        growth_stage_time = growth_stage_time,
        cultivation_temp = cultivation_temp,
        species = "http://purl.obolibrary.org/obo/NCBITaxon_6239",
        sex = sex,
        strain = strain
    )

    return nwbfile

def create_device(nwbfile, name, description, manufacturer):

    device = nwbfile.create_device(
        name = name,
        description = description,
        manufacturer = manufacturer
    )

    return device


def create_im_vol(nwbfile, name, device, description, channels, location="head", grid_spacing=[0.3208, 0.3208, 0.75], grid_spacing_unit ="micrometers", origin_coords=[0,0,0], origin_coords_unit="micrometers", reference_frame="Worm head"):
    
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
        OptChanRefData.append(fluor)

    OptChanRefs = OpticalChannelReferences(
        name = 'order_optical_channels',
        channels = OptChanRefData
    )

    ImVol = ImagingVolume(
        name= name,
        optical_channel_plus = OptChannels,
        order_optical_channels = OptChanRefs,
        description = description,
        device = device,
        location = location,
        grid_spacing = grid_spacing,
        grid_spacing_unit = grid_spacing_unit,
        origin_coords = origin_coords,
        origin_coords_unit = origin_coords_unit,
        reference_frame = reference_frame
    )

    nwbfile.add_imaging_plane(ImVol)

    return ImVol

def create_image(name, description, data, ImVol, RGBW_channels=[0,1,2,3]):

    image = MultiChannelVolume(
        name = name,
        description = description,
        RGBW_channels = RGBW_channels,
        data = data,
        imaging_volume = ImVol
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

    vs = PlaneSegmentation(
        name = name,
        description = description,
        imaging_plane = ImagingVolume,
        reference_images = reference_images
    )

    for i in range(positions.shape[0]):
        voxel_mask = []
        x = positions[i,0]
        y = positions[i,1]
        z = positions[i,2]

        voxel_mask.append([np.uint(x),np.uint(y),np.uint(z),1])  # add weight of 1 to each ROI

        vs.add_roi(voxel_mask=voxel_mask)

    if labels is None:
        labels = ['']*positions.shape[0]
    else:
        vs.add_column(
            name = 'ID_labels',
            description = 'ROI ID labels',
            data = labels,
            index=True
        )
    
    return vs

def create_calc_series(name, data, description, comments,  device, imaging_volume, unit, scan_line_rate, dimension, rate, resolution, compression = False):

    if compression:
        data = H5DataIO(data=data, compression="gzip", compression_opts=4)

    calcium_image_series = MultiChannelVolumeSeries(
        name = name,
        data = data,
        description = description,
        comments = comments,
        device = device,
        unit = unit,
        scan_line_rate = scan_line_rate,
        dimension = dimension,
        rate = rate,
        resolution= resolution,
        imaging_volume = imaging_volume
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

def process_NP_FOCO_original(datapath,dataset, strain, calc = False):

    identifier = dataset
    session_description = 'NeuroPAL and calcium imaging of immobilized worm with optogenetic stimulus'
    session_start_time = datetime(int(identifier[0:4]), int(identifier[5:7]), int(identifier[8:10]), tzinfo=tz.gettz("US/Pacific"))
    lab = 'FOCO lab'
    institution = 'UCSF'
    pubs = ''
    experimenter = 'Jackson Borchardt + Greg Bubnis'
    if calc:
        experiment_description = 'NeuroPAL images and GCaMP time series'
    else:
        experiment_description = 'NeuroPAL structural images'

    nwbfile = gen_file(session_description, experimenter, experiment_description, identifier, session_start_time, lab, institution, pubs)

    subject_description = 'NeuroPAL worm in microfluidic chip'
    dob = datetime(int(identifier[0:4]), int(identifier[5:7]), int(identifier[8:10]), tzinfo=tz.gettz("US/Pacific"))
    growth_stage = 'YA'
    gs_time = pd.Timedelta(hours=2, minutes=30).isoformat()
    cultivation_temp = 20.
    sex = "O"

    nwbfile = create_subject(nwbfile, subject_description, identifier, dob, growth_stage, gs_time, cultivation_temp, sex, strain)

    microname = "Spinning disk confocal"
    microdescrip = "Leica DMi8 Inverted Microscope with Yokogawa CSU-W1 SoRA, 40x WI objective 1.1 NA"
    manufacturer = "Leica, Yokogawa"

    microscope = create_device(nwbfile, microname, microdescrip, manufacturer)

    matfile = datapath + '/NP_FOCO_cropped/' +dataset +'/neuropal_1_MMStack_Pos0.ome.mat'
    mat = sio.loadmat(matfile)
    scale = np.asarray(mat['info']['scale'][0][0]).flatten()

    channels = [("mNeptune 2.5", "Chroma ET 700/75", "561-700-75m"), ("Tag RFP-T", "Chroma ET 605/70", "561-605-70m"), ("CyOFP1", "Chroma ET 605/70","488-605-70m"), ("GFP-GCaMP", "Chroma ET 525/50","488-525-50m"), ("mTagBFP2", "Chroma ET 460/50", "405-460-50m"),  ("mNeptune 2.5-far red", "Chroma ET 700/75", "639-700-75m")]
    RGBW_channels = [0,2,4,1]

    NP_descrip = 'NeuroPAL image of C. elegans brain'

    NP_ImVol = create_im_vol(nwbfile, 'NeuroPALImVol', microscope, NP_descrip,channels, location="head", grid_spacing = scale)

    raw_file = datapath + '/NP_FOCO_cropped/' + dataset + '/neuropal_1_MMStack_Pos0.ome.tif'
    data = skio.imread(raw_file)
    data = np.transpose(data)

    ImDescrip = 'NeuroPAL structural image'

    NP_image = create_image('NeuroPALImageRaw', ImDescrip, data, NP_ImVol, RGBW_channels=RGBW_channels)

    nwbfile.add_acquisition(NP_image)

    blob_file = datapath + '/NP_FOCO_cropped/' + dataset + '/blobs.csv'
    blobs = pd.read_csv(blob_file)

    IDs = blobs['ID']
    labels = IDs.replace(np.nan,'',regex=True)
    labels = list(np.asarray(labels)) 
    positions = np.asarray(blobs[['X', 'Y', 'Z']])

    vs_descrip = 'Neuron centers for multichannel volumetric image. Weight set at 1 for all voxels. Labels refers to cell ID of segmented neurons.'

    NeuroPALImSeg = ImageSegmentation(
        name = 'NeuroPALSegmentation',
        plane_segmentations = create_vol_seg_centers('NeuroPALNeurons', vs_descrip, NP_ImVol, positions, labels=labels)
    )

    neuroPAL_module = nwbfile.create_processing_module(
        name = 'NeuroPAL',
        description = 'NeuroPAL image data and metadata'
    )

    neuroPAL_module.add(NeuroPALImSeg)

    Proc_descrip = 'Processed NeuroPAL image'

    Proc_ImVol = create_im_vol(nwbfile, 'ProcessedImVol', microscope, Proc_descrip,[channels[i] for i in RGBW_channels])

    if not os.path.exists(datapath+ '/NP_FOCO_hist_match/' + dataset + '/hist_match_image.tif'):
        proc_mat = datapath+ '/NP_FOCO_hist_match/' + dataset + '/hist_match_image.mat'

        imfile = sio.loadmat(proc_mat)
        im = np.transpose(imfile['Hist_RGBW'],(2,3,0,1))
        im = im.astype('uint16')

        tifffile.imwrite(datapath + '/NP_FOCO_hist_match/'+folder+'/hist_match_image.tif', im, imagej = True)

    proc_file = datapath+ '/NP_FOCO_hist_match/' + dataset + '/hist_match_image.tif'
    proc_data = np.transpose(skio.imread(proc_file), [2,1,0,3])

    ProcDescrip = 'NeuroPAL image with median filtering followed by color histogram matching to reference NeuroPAL images'

    Proc_image = create_image('ProcessedImage', ProcDescrip, proc_data, Proc_ImVol, RGBW_channels=[0,1,2,3])

    processed_im_module = nwbfile.create_processing_module(
        name = 'ProcessedImage',
        description = 'Data and metadata associated with the pre-processed neuroPAL image.'
    )

    processed_im_module.add(Proc_image)

    if calc:

        GCaMP_chan = [("GFP-GCaMP", "Chroma ET 525/50","488-525-50m")]

        Calc_scale = [0.3208, 0.3208, 1.5]

        Calc_descrip = 'Imaging volume used to acquire calcium imaging data'

        Calc_ImVol = create_im_vol(nwbfile, 'CalciumImVol', microscope, Calc_descrip, GCaMP_chan, grid_spacing=Calc_scale)

        Calc_file = datapath + '/NP_FOCO_cropped/' + dataset +'/' +dataset+'.tiff'

        tif = TiffFile(Calc_file)

        page = tif.pages[0]
        numx = page.shape[0]
        numy = page.shape[1]
        numz = 12

        data = DataChunkIterator(
            data = iter_calc_tiff(Calc_file, numz),
            maxshape = None,
            buffer_size = 10
        )

        Calc_name = 'CalciumImageSeries'
        description = 'Raw GCaMP series images'
        comments = 'single channel GFP-GCaMP representing GCaMP signal'
        Calc_unit = "Voxel gray counts"
        scan_line_rate = 9504.
        rate = 3.3
        resolution = 1.0

        Calc_ImSeries = create_calc_series(Calc_name, data, description, comments, microscope,Calc_ImVol, Calc_unit, scan_line_rate, [numx, numy, numz], rate, resolution, compression=True)

        nwbfile.add_acquisition(Calc_ImSeries)

        gce_file = datapath + '/NP_FOCO_cropped/' + dataset +'/' + dataset + '_dff.csv'

        gce_quant = pd.read_csv(gce_file)

        gce_df = gce_quant[['x', 'y', 'z', 'dff', 'neuron_id', 't', 'neuron_ndx']]

        blobquant = None
        for idx in gce_quant['neuron_ndx'].unique():
            blob = gce_df[gce_df['neuron_ndx']==idx]
            blobarr = np.asarray(blob[['x','y','z','dff','neuron_id']]) 
            blobarr = blobarr[np.newaxis, :, :]
            if blobquant is None:
                blobquant=blobarr

            else:
                blobquant = np.vstack((blobquant, blobarr))

        volsegs = []

        for t in range(blobquant.shape[1]):
            blobs = np.squeeze(blobquant[:,t,0:3])
            IDs = np.squeeze(blobquant[:,t,4])
            labels = IDs.astype(str)
            labels = np.where(labels!='nan', labels, '')

            vsname = 'Seg_tpoint_'+str(t)
            description = 'Neuron segmentation for time point ' +str(t) + ' in calcium image series'
            volseg = create_vol_seg_centers(vsname, description, Calc_ImVol, blobs, labels=labels)

            volsegs.append(volseg)


        CalcImSeg = ImageSegmentation(
            name = 'CalciumSeriesSegmentation',
            plane_segmentations = volsegs
        )

        calc_IDs = np.squeeze(blobquant[:,0,4])
        calc_labels = calc_IDs.astype(str)
        calc_labels = np.where(calc_labels!='nan', calc_labels, '')

        Calclabels = SegmentationLabels(
            name = 'NeuronIDs',
            labels = calc_labels,
            description = 'Calcium ROI segmentation labels',
            ImageSegmentation = CalcImSeg
        )

        gce_data = np.transpose(blobquant[:,:,3])
        gce_data = gce_data.astype(float)

        rt_region = volsegs[0].create_roi_table_region(
            description = 'Segmented neurons associated with calcium image series. This rt_region uses the location of the neurons at the first time point',
            region = list(np.arange(blobquant.shape[0]))
        )

        RoiResponse = RoiResponseSeries( # CHANGE WITH FEEDBACK FROM RAY
            name = 'SignalCalciumImResponseSeries',
            description = 'DF/F activity for calcium imaging data',
            data = gce_data,
            rois = rt_region,
            unit = 'Percentage',
            rate = 3.3
        )

        SignalFluor = DfOverF(
            name = 'SignalDFoF',
            roi_response_series = RoiResponse
        )

        calcium_im_module = nwbfile.create_processing_module(
        name = 'CalciumActivity',
        description = 'Data and metadata associated with time series of calcium images'
        )

        calcium_im_module.add(CalcImSeg)
        calcium_im_module.add(SignalFluor)
        calcium_im_module.add(Calclabels)


    io = NWBHDF5IO(datapath + '/final_nwb/SK2/'+identifier+'.nwb', mode='w')
    io.write(nwbfile)
    io.close()


def process_NP_FOCO_Ray(datapath, dataset, strain, calc=False):

    identifier = dataset
    session_description = 'NeuroPAL and calcium imaging of immobilized worm with optogenetic stimulus'
    session_start_time = datetime(int(identifier[0:4]), int(identifier[4:6]), int(identifier[6:8]), int(identifier[9:11]), int(identifier[12:14]), int(identifier[15:]), tzinfo=tz.gettz("US/Pacific"))
    lab = 'FOCO lab'
    institution = 'UCSF'
    pubs = ''
    experimenter = 'Raymond Dunn'
    experiment_description = 'optogenetic stimulation of single neurons'

    nwbfile = gen_file(session_description, experimenter, experiment_description, identifier, session_start_time, lab, institution, pubs)

    subject_description = 'NeuroPAL worm in microfluidic chip'
    dob = datetime(int(identifier[0:4]), int(identifier[4:6]), int(identifier[6:8]), tzinfo=tz.gettz("US/Pacific"))
    growth_stage = 'YA'
    gs_time = pd.Timedelta(hours=2, minutes=30).isoformat()
    cultivation_temp = 20.
    sex = "O"

    nwbfile = create_subject(nwbfile, subject_description, identifier, dob, growth_stage, gs_time, cultivation_temp, sex, strain)

    microname = "Spinning disk confocal"
    microdescrip = "Leica DMi8 Inverted Microscope with Yokogawa CSU-W1 SoRA, 40x WI objective 1.1 NA"
    manufacturer = "Leica, Yokogawa"

    microscope = create_device(nwbfile, microname, microdescrip, manufacturer)

    matfile = datapath + '/Manual_annotate/' +dataset +'/neuroPAL_image.mat'
    mat = sio.loadmat(matfile)
    scale = np.asarray(mat['info']['scale'][0][0]).flatten()


    #TODO: update if statement for 0426 datasets
    if folder <'20230322':
        channels = [("mTagBFP2", "Chroma ET 460/50", "405-460-50m"), ("CyOFP1", "Chroma ET 605/70","488-605-70m"), ("GFP-GCaMP", "Chroma ET 525/50","488-525-50m"), ("mNeptune 2.5", "Chroma ET 700/75", "561-700-75m"), ("Tag RFP-T", "Chroma ET 605/70", "561-605-70m"), ("mNeptune 2.5-far red", "Chroma ET 700/75", "639-700-75m")]
        RGBW_channels = [3,1,0,4]
    else:
        channels = [("mTagBFP2", "Chroma ET 460/50", "405-460-50m"), ("CyOFP1", "Chroma ET 605/70","488-605-70m"), ("CyOFP1-high filter", "Chroma ET 700/75","488-700-75m"), ("GFP-GCaMP", "Chroma ET 525/50","488-525-50m"), ("mNeptune 2.5", "Chroma ET 700/75", "561-700-75m"), ("Tag RFP-T", "Chroma ET 605/70", "561-605-70m"), ("mNeptune 2.5-far red", "Chroma ET 700/75", "639-700-75m")]
        RGBW_channels = [4,1,0,6]

    NP_descrip = 'NeuroPAL image of C. elegans brain'

    NP_ImVol = create_im_vol(nwbfile, 'NeuroPALImVol', microscope, NP_descrip,channels, location="head", grid_spacing = scale)

    raw_file = datapath + '/NP_Ray/' + dataset + '/full_comp.tif'
    data = skio.imread(raw_file)
    data = np.transpose(data)

    ImDescrip = 'NeuroPAL structural image'

    NP_image = create_image('NeuroPALImageRaw', ImDescrip, data, NP_ImVol, RGBW_channels=RGBW_channels)

    nwbfile.add_acquisition(NP_image)

    blob_file = datapath + '/Manual_annotate/' + dataset + '/blobs.csv'
    blobs = pd.read_csv(blob_file)

    IDs = blobs['ID']
    labels = IDs.replace(np.nan,'',regex=True)
    labels = list(np.asarray(labels)) 
    positions = np.asarray(blobs[['X', 'Y', 'Z']])

    vs_descrip = 'Neuron centers for multichannel volumetric image. Weight set at 1 for all voxels. Labels refers to cell ID of segmented neurons.'

    NeuroPALImSeg = ImageSegmentation(
        name = 'NeuroPALSegmentation',
        plane_segmentations = create_vol_seg_centers('NeuroPALNeurons', vs_descrip, NP_ImVol, positions, labels=labels)
    )

    neuroPAL_module = nwbfile.create_processing_module(
        name = 'NeuroPAL',
        description = 'NeuroPAL image data and metadata'
    )

    neuroPAL_module.add(NeuroPALImSeg)

    Proc_descrip = 'Processed NeuroPAL image'

    Proc_ImVol = create_im_vol(nwbfile, 'ProcessedImVol', microscope, Proc_descrip,[channels[i] for i in RGBW_channels])

    proc_file = datapath+ '/manual_annotate/' + dataset + '/neuroPAL_image.tif'
    proc_data = np.transpose(skio.imread(proc_file), [2,1,0,3])

    ProcDescrip = 'NeuroPAL image with median filtering followed by color histogram matching to reference NeuroPAL images'

    Proc_image = create_image('ProcessedImage', ProcDescrip, proc_data, Proc_ImVol, RGBW_channels=[0,1,2,3])

    processed_im_module = nwbfile.create_processing_module(
        name = 'ProcessedImage',
        description = 'Data and metadata associated with the pre-processed neuroPAL image.'
    )

    processed_im_module.add(Proc_image)

    if calc:
        GCaMP_chan = [("GFP-GCaMP", "Chroma ET 525/50","488-525-50m")]

        if folder <'20230506':
            Calc_scale = [0.3208, 0.3208, 2.5]
        else:
            Calc_scale = [0.1604, 0.1604, 3.0]

        Calc_descrip = 'Imaging volume used to acquire calcium imaging data'

        Calc_ImVol = create_im_vol(nwbfile, 'CalciumImVol', microscope, Calc_descrip, GCaMP_chan, grid_spacing=Calc_scale)

        Calc_file = datapath + '/NP_Ray/' + dataset +'/' +dataset+'.tiff'

        tif = TiffFile(Calc_file)

        page = tif.pages[0]
        numx = page.shape[0]
        numy = page.shape[1]
        numz = 12

        data = DataChunkIterator(
            data = iter_calc_tiff(Calc_file, numz),
            maxshape = None,
            buffer_size = 10
        )

        Calc_name = 'CalciumImageSeries'
        description = 'Raw GCaMP series images'
        comments = 'single channel GFP-GCaMP representing GCaMP signal'
        Calc_unit = "Voxel gray counts"
        scan_line_rate = 2995.
        rate = 1.04
        resolution = 1.0

        Calc_ImSeries = create_calc_series(Calc_name, data, description, comments, microscope,Calc_ImVol, Calc_unit, scan_line_rate, [numx, numy, numz], rate, resolution, compression=True)

        nwbfile.add_acquisition(Calc_ImSeries)

        gce_file = datapath + '/NP_Ray/' + dataset +'/' + dataset + '_dff.csv'

        gce_quant = pd.read_csv(gce_file)

        gce_df = gce_quant[['x', 'y', 'z', 'dff', 'neuron_id', 't', 'neuron_ndx']]

        blobquant = None
        for idx in gce_quant['neuron_ndx'].unique():
            blob = gce_df[gce_df['neuron_ndx']==idx]
            blobarr = np.asarray(blob[['x','y','z','dff','neuron_id']]) 
            blobarr = blobarr[np.newaxis, :, :]
            if blobquant is None:
                blobquant=blobarr

            else:
                blobquant = np.vstack((blobquant, blobarr))

        volsegs = []

        for t in range(blobquant.shape[1]):
            blobs = np.squeeze(blobquant[:,t,0:3])
            IDs = np.squeeze(blobquant[:,t,4])
            labels = IDs.astype(str)
            labels = np.where(labels!='nan', labels, '')

            vsname = 'Seg_tpoint_'+str(t)
            description = 'Neuron segmentation for time point ' +str(t) + ' in calcium image series'
            volseg = create_vol_seg_centers(vsname, description, Calc_ImVol, blobs, labels=labels)

            volsegs.append(volseg)


        CalcImSeg = ImageSegmentation(
            name = 'CalciumSeriesSegmentation',
            plane_segmentations = volsegs
        )

        calc_IDs = np.squeeze(blobquant[:,0,4])
        calc_labels = calc_IDs.astype(str)
        calc_labels = np.where(calc_labels!='nan', calc_labels, '')

        Calclabels = SegmentationLabels(
            name = 'NeuronIDs',
            labels = calc_labels,
            description = 'Calcium ROI segmentation labels',
            ImageSegmentation = CalcImSeg
        )

        gce_data = np.transpose(blobquant[:,:,3])
        gce_data = gce_data.astype(float)

        rt_region = volsegs[0].create_roi_table_region(
            description = 'Segmented neurons associated with calcium image series. This rt_region uses the location of the neurons at the first time point',
            region = list(np.arange(blobquant.shape[0]))
        )

        RoiResponse = RoiResponseSeries( # CHANGE WITH FEEDBACK FROM RAY
            name = 'SignalCalciumImResponseSeries',
            description = 'DF/F activity for calcium imaging data',
            data = gce_data,
            rois = rt_region,
            unit = 'Percentage',
            rate = 1.04
        )

        SignalFluor = DfOverF(
            name = 'SignalDFoF',
            roi_response_series = RoiResponse
        )

        calcium_im_module = nwbfile.create_processing_module(
        name = 'CalciumActivity',
        description = 'Data and metadata associated with time series of calcium images'
        )

        calcium_im_module.add(CalcImSeg)
        calcium_im_module.add(SignalFluor)
        calcium_im_module.add(Calclabels)

    io = NWBHDF5IO(datapath + '/final_nwb/SK1/'+identifier+'.nwb', mode='w')
    io.write(nwbfile)
    io.close()

def process_yemini(folder):
    worm = folder.split('/')[-1]

    matfile = folder + '/head.mat'
    csvfile = folder + '/head.csv'
    gcampfile = folder + '/gcamp.mat'
    gclabelfile = folder + '/gclabels.mat'
    activity = folder + '/activity.mat'
    positions = folder + '/positions.mat'
    oldact = folder +'/old_act.mat'
    oldpos = folder +'/old_pos.mat'

    mat = sio.loadmat(matfile)
    gcamp = sio.loadmat(gcampfile)

    data = np.transpose(mat['data']*4095, (1,0,2,3))

    gcdata = gcamp['data']

    activitydata = sio.loadmat(activity)['neuron_activity']
    positiondata = sio.loadmat(positions)['neuron_positions']
    oldact = sio.loadmat(oldact)['old_neuron_activity']
    oldpos = sio.loadmat(oldpos)['old_neuron_positions']
    gclabels = sio.loadmat(gclabelfile)['gclabels']

    gclabels = np.asarray([label.replace(" ","") for label in gclabels])

    gcdata = np.transpose(gcdata, (3,1,0,2)) # convert data to TXYZ

    ydim = gcdata.shape[2]
    zdim = gcdata.shape[3]

    scale = np.asarray(mat['info']['scale'][0][0]).flatten()
    prefs = np.asarray(mat['prefs']['RGBW'][0][0]).flatten()-1 #subtract 1 to adjust for matlab indexing from 1
    
    gcscale = np.asarray(gcamp['worm_data']['info'][0][0][0][0][1]).flatten()

    session_start = datetime(int(worm[0:4]),int(worm[4:6]),int(worm[6:8]), tzinfo=tz.gettz("US/Pacific"))

    experimenter = 'Yemini, Eviatar'
    experiment_descrip = 'NeuroPAL and whole-brain calcium imaging with chemical stimuli'

    nwbfile = gen_file('C. elegans head NeuroPAL and Calcium imaging', experimenter, experiment_descrip, worm, session_start, 'Hobert lab', 'Columbia University', ["NeuroPAL: A Multicolor Atlas for Whole-Brain Neuronal Identification in C. elegans", "Extracting neural signals from semi-immobilized animals with deformable non-negative matrix factorization" ])

    subject_description = 'NeuroPAL worm in microfluidic chip'
    dob = session_start - timedelta(days=2)
    growth_stage = 'YA'
    gs_time = None
    cultivation_temp = 20.
    sex = "O"
    strain = "OH16230"

    nwbfile = create_subject(nwbfile, subject_description, worm, dob, growth_stage, gs_time, cultivation_temp, sex, strain)

    microname = "Spinning disk confocal"
    microdescrip = "Spinning Disk Confocal Nikon Ti-e 60x Objective, 1.2 NA	Nikon CFI Plan Apochromat VC 60XC WI"
    manufacturer = "Nikon"

    device = create_device(nwbfile, microname, microdescrip, manufacturer)

    if prefs[3]== 4:
        channels = [("mTagBFP2", "Semrock FF01-445/45-25 Brightline", "405-445-45m"), ("CyOFP1", "Semrock FF02-617/73-25 Brightline", "488-610-40m"), ("mNeptune 2.5", "Semrock FF01-731/137-25 Brightline","561-731-70m"), ("GFP-GCaMP", "Semrock FF02-525/40-25 Brightline", "488-525-25m"), ("Tag RFP-T", "Semrock FF02-617/73-25 Brightline", "561-610-40m")]
    elif prefs[3]==3:
        channels = [("mTagBFP2", "Semrock FF01-445/45-25 Brightline", "405-445-25m"), ("CyOFP1", "Semrock FF02-617/73-25 Brightline","488-610-40m"), ("mNeptune 2.5", "Semrock FF01-731/137-25 Brightline","561-731-70m"), ("Tag RFP-T", "Semrock FF02-617/73-25 Brightline", "561-610-40m"), ("GFP-GCaMP", "Semrock FF02-525/40-25 Brightline", "488-525-25m")]

    NP_ImVol = create_im_vol(nwbfile, 'NeuroPALImVol', device, 'NeuroPAL image of C. elegans brain', channels, grid_spacing=scale)

    csv = pd.read_csv(csvfile, skiprows=6)

    blobs = csv[['Real X (um)', 'Real Y (um)', 'Real Z (um)', 'User ID']]
    blobs = blobs.rename(columns={'Real X (um)':'X', 'Real Y (um)':'Y', 'Real Z (um)':'Z', 'User ID':'ID'})
    blobs['X'] = round(blobs['X'].div(scale[0])) -1
    blobs['Y'] = round(blobs['Y'].div(scale[1])) -1
    blobs['Z'] = round(blobs['Z'].div(scale[2])) -1
    blobs = blobs.astype({'X':'uint16', 'Y':'uint16', 'Z':'uint16'})
    pos = np.asarray(blobs[['X', 'Y', 'Z']])
    IDs = blobs['ID']
    labels = IDs.replace(np.nan, '', regex=True)
    labels = list(np.asarray(labels))

    vs_descrip = 'Neuron centers for multichannel volumetric image. Weight set at 1 for all voxels. Labels refers to cell ID of segmented neurons.'

    NeuroPALImSeg = ImageSegmentation(
        name = 'NeuroPALSegmentation',
        plane_segmentations = create_vol_seg_centers('NeuroPALNeurons', vs_descrip, NP_ImVol, pos, labels)
    )

    neuroPAL_module = nwbfile.create_processing_module(
        name = 'NeuroPAL',
        description = 'NeuroPAL image data and metadata'
    )

    neuroPAL_module.add(NeuroPALImSeg)

    NP_descrip = 'NeuroPAL structural image'

    NP_image = create_image('NeuroPALImageRaw', NP_descrip, data, NP_ImVol, RGBW_channels=prefs)

    nwbfile.add_acquisition(NP_image)

    gc_optchan = [("GFP-GCaMP", "Semrock FF02-525/40-25 Brightline","488-525-25m")]

    Calc_descrip = 'Imaging volume used to acquire calcium imaging data'

    Calc_ImVol = create_im_vol(nwbfile, 'CalciumImVol', device, Calc_descrip, gc_optchan, grid_spacing=gcscale)

    Calc_name = 'CalciumImageSeries'
    description = 'Raw GCaMP series images'
    comments = 'single channel GFP-GCaMP representing GCaMP signal'
    Calc_unit = "Voxel gray counts"
    rate = 4.0
    scan_line_rate = float(ydim*zdim*rate)
    resolution = 1.0
    dimensions = gcdata.shape

    Calc_ImSeries = create_calc_series(Calc_name, gcdata, description, comments, device, Calc_ImVol, Calc_unit, scan_line_rate, [dimensions[1], dimensions[2], dimensions[3]], rate, resolution, compression=False)

    nwbfile.add_acquisition(Calc_ImSeries)

    positions = positiondata[:,:,[1,0,2]] #switch first two columns so that x is first in accordance with image data
    raw_pos = oldpos[:,:,[1,0,2]]

    volsegs = []

    for t in range(raw_pos.shape[1]):
        blobs = np.squeeze(raw_pos[:,t,:])

        vsname = 'Seg_tpoint_'+str(t)
        description = 'Neuron segmentation for time point ' +str(t) + ' in calcium image series'
        volseg = create_vol_seg_centers(vsname, description, Calc_ImVol, blobs)

        volsegs.append(volseg)

    CalcImSeg = ImageSegmentation(
        name = 'CalciumSeriesSegmentation',
        plane_segmentations = volsegs
    )

    procvolsegs = []
    for t in range(positions.shape[1]):
        blobs = np.squeeze(positions[:,t,:])

        vsname = 'Seg_tpoint_'+str(t)
        description = 'Neuron positions for time point ' +str(t) + ' calculated via dNMF'
        volseg = create_vol_seg_centers(vsname, description, Calc_ImVol, blobs)

        procvolsegs.append(volseg)

    ProcCalcImSeg = ImageSegmentation(
        name = 'CalciumSeriesSegmentationdNMF',
        plane_segmentations = procvolsegs
    )

    calcium_labels = SegmentationLabels(
        name = 'NeuronIDs',
        labels = gclabels,
        description = 'Neuron labels for calcium activity traces',
        ImageSegmentation = CalcImSeg
    )

    gce_data = np.transpose(activitydata)

    rt_region = volsegs[0].create_roi_table_region(
        description = 'Segmented neurons associated with calcium image series. This rt_region uses the location of the neurons at the first time point',
        region = list(np.arange(positions.shape[0]))
    )

    RoiResponse = RoiResponseSeries( # CHANGE WITH FEEDBACK FROM RAY
        name = 'SignalCalciumImResponseSeries',
        description = 'Raw fluorescence activity for calcium imaging data',
        data = gce_data,
        rois = rt_region,
        unit = 'unitless',
        rate = 4.0
    )

    SignalFluor = Fluorescence(
        name = 'SignalRawFluor',
        roi_response_series = RoiResponse
    )

    RoiResponsedNMF = RoiResponseSeries( # CHANGE WITH FEEDBACK FROM RAY
        name = 'dNMFCalciumImResponseSeries',
        description = 'Raw fluorescence activity data computed using dNMF',
        data = gce_data,
        rois = rt_region,
        unit = 'unitless',
        rate = 4.0
    )

    dNMFFluor = Fluorescence(
        name = 'SignalRawFluordNMF',
        roi_response_series = RoiResponsedNMF
    )

    calcium_im_module = nwbfile.create_processing_module(
    name = 'CalciumActivity',
    description = 'Data and metadata associated with time series of calcium images'
    )

    calcium_im_module.add(CalcImSeg)
    calcium_im_module.add(ProcCalcImSeg)
    calcium_im_module.add(SignalFluor)
    calcium_im_module.add(dNMFFluor)
    calcium_im_module.add(calcium_labels)

    io = NWBHDF5IO(datapath + '/final_nwb/EY/'+worm+'.nwb', mode='w')
    io.write(nwbfile)
    io.close()

def process_Yem_original(file):

    worm = file.split('.')[0]

    path = '/Users/danielysprague/foco_lab/data/NP_paper/all/'
    matfile = path + file
    ID_file = path + worm +'_ID.mat'
    csvfile = path + worm + '.csv'

    mat = sio.loadmat(matfile)

    data = np.transpose(mat['data'], (1,0,2,3))

    scale = np.asarray(mat['info']['scale'][0][0]).flatten()
    prefs = np.asarray(mat['prefs']['RGBW'][0][0]).flatten()-1 #subtract 1 to adjust for matlab indexing from 1

    session_start = datetime(2021,1,7, tzinfo=tz.gettz("US/Pacific")) #currently just using date of paper publication

    experimenter = 'Yemini, Eviatar'
    experiment_descrip = 'NeuroPAL whole-brain structural images'

    nwbfile = gen_file('C. elegans head NeuroPAL', experimenter, experiment_descrip, worm, session_start, 'Hobert lab', 'Columbia University', ["NeuroPAL: A Multicolor Atlas for Whole-Brain Neuronal Identification in C. elegans"])

    subject_description = 'NeuroPAL worm in microfluidic chip'
    dob = session_start - timedelta(days=2)
    growth_stage = 'YA'
    gs_time = None
    cultivation_temp = 20.
    sex = "O"
    strain = "OH16230"

    nwbfile = create_subject(nwbfile, subject_description, worm, dob, growth_stage, gs_time, cultivation_temp, sex, strain)

    microname = "Spinning disk confocal"
    microdescrip = "Spinning Disk Confocal Nikon Ti-e 60x Objective, 1.2 NA	Nikon CFI Plan Apochromat VC 60XC WI"
    manufacturer = "Nikon"

    device = create_device(nwbfile, microname, microdescrip, manufacturer)

    channels = [("mTagBFP2", "Semrock FF01-445/45-25 Brightline", "405-445-45m"), ("Tag RFP-T", "Semrock FF02-617/73-25 Brightline", "561-610-40m"), ("GFP-GCaMP", "Semrock FF02-525/40-25 Brightline", "488-525-25m"), ("CyOFP1", "Semrock FF02-617/73-25 Brightline", "488-610-40m"),  ("mNeptune 2.5", "Semrock FF01-731/137-25 Brightline","561-731-70m")]

    NP_ImVol = create_im_vol(nwbfile, 'NeuroPALImVol', device, 'NeuroPAL image of C. elegans brain', channels, grid_spacing=scale)

    csv = pd.read_csv(csvfile, skiprows=6)

    blobs = csv[['Real X (um)', 'Real Y (um)', 'Real Z (um)', 'User ID']]
    blobs = blobs.rename(columns={'Real X (um)':'X', 'Real Y (um)':'Y', 'Real Z (um)':'Z', 'User ID':'ID'})
    blobs['X'] = round(blobs['X'].div(scale[0])) -1
    blobs['Y'] = round(blobs['Y'].div(scale[1])) -1
    blobs['Z'] = round(blobs['Z'].div(scale[2])) -1
    blobs = blobs.astype({'X':'uint16', 'Y':'uint16', 'Z':'uint16'})
    pos = np.asarray(blobs[['X', 'Y', 'Z']])
    IDs = blobs['ID']
    labels = IDs.replace(np.nan, '', regex=True)
    labels = list(np.asarray(labels))

    vs_descrip = 'Neuron centers for multichannel volumetric image. Weight set at 1 for all voxels. Labels refers to cell ID of segmented neurons.'

    NeuroPALImSeg = ImageSegmentation(
        name = 'NeuroPALSegmentation',
        plane_segmentations = create_vol_seg_centers('NeuroPALNeurons', vs_descrip, NP_ImVol, pos, labels)
    )

    neuroPAL_module = nwbfile.create_processing_module(
        name = 'NeuroPAL',
        description = 'NeuroPAL image data and metadata'
    )

    neuroPAL_module.add(NeuroPALImSeg)

    NP_descrip = 'NeuroPAL structural image'

    NP_image = create_image('NeuroPALImageRaw', NP_descrip, data, NP_ImVol, RGBW_channels=prefs)

    nwbfile.add_acquisition(NP_image)

    io = NWBHDF5IO(datapath + '/final_nwb/NP_og/'+worm+'.nwb', mode='w')
    io.write(nwbfile)
    io.close()

def process_chaudhary(folder):

    matfile = folder + '/Composite.mat'
    csvfile = folder + '/mark_w_names.csv'

    worm = folder[-1]

    mat = sio.loadmat(matfile)

    data = np.transpose(mat['data'], (1,0,2,3))

    scale = np.asarray([0.33, 0.33, 1])
    prefs = np.asarray(mat['prefs']['RGBW'][0][0]).flatten()-1 #subtract 1 to adjust for matlab indexing from 1

    session_start = datetime(2021,2,24, tzinfo=tz.gettz("US/Pacific")) #currently just using date of paper publication

    experimenter = 'Chaudhary, Shivesh'
    experiment_descrip = 'NeuroPAL whole-brain structural images'

    nwbfile = gen_file('C. elegans head NeuroPAL', experimenter, experiment_descrip, worm, session_start, 'Lu Lab', 'Georgia Tech University', ["Graphical-model framework for automated annotation of cell identities in dense cellular images"])

    subject_description = 'NeuroPAL worm in microfluidic chip'
    dob = session_start - timedelta(days=2)
    growth_stage = 'YA'
    gs_time = None
    cultivation_temp = 20.
    sex = "O"
    strain = "OH15495"

    nwbfile = create_subject(nwbfile, subject_description, worm, dob, growth_stage, gs_time, cultivation_temp, sex, strain)

    microname = "Spinning disk confocal"
    microdescrip = "Perkin Elmer spinning disk confocal microscope 40x Objective, oil objective with an EMCCD camera"
    manufacturer = "Perkin Elmer"

    device = create_device(nwbfile, microname, microdescrip, manufacturer)

    channels = [("mNeptune 2.5", "Semrock FF01-731/137-25 Brightline","561-731-70m"), ("CyOFP1", "Semrock FF02-617/73-25 Brightline", "488-610-40m"), ("mTagBFP2", "Semrock FF01-445/45-25 Brightline", "405-445-45m"), ("Tag RFP-T", "Semrock FF02-617/73-25 Brightline", "561-610-40m")]

    NP_ImVol = create_im_vol(nwbfile, 'NeuroPALImVol', device, 'NeuroPAL image of C. elegans brain', channels, grid_spacing=scale)

    csv = pd.read_csv(csvfile)

    blobs = csv[['##x', 'y', 'z', 'ID']]
    blobs = blobs.rename(columns={'##x':'X', 'y':'Y', 'z':'Z', 'ID':'ID'})
    blobs['X'] = round(blobs['X'])
    blobs['Y'] = round(blobs['Y'])
    blobs['Z'] = round(blobs['Z'])
    blobs = blobs.astype({'X':'uint16', 'Y':'uint16', 'Z':'uint16'})
    pos = np.asarray(blobs[['X', 'Y', 'Z']])
    IDs = blobs['ID']
    labels = IDs.replace(np.nan, '', regex=True)
    labels = list(np.asarray(labels))

    vs_descrip = 'Neuron centers for multichannel volumetric image. Weight set at 1 for all voxels. Labels refers to cell ID of segmented neurons.'

    NeuroPALImSeg = ImageSegmentation(
        name = 'NeuroPALSegmentation',
        plane_segmentations = create_vol_seg_centers('NeuroPALNeurons', vs_descrip, NP_ImVol, pos, labels)
    )

    neuroPAL_module = nwbfile.create_processing_module(
        name = 'NeuroPAL',
        description = 'NeuroPAL image data and metadata'
    )

    neuroPAL_module.add(NeuroPALImSeg)

    NP_descrip = 'NeuroPAL structural image'

    NP_image = create_image('NeuroPALImageRaw', NP_descrip, data, NP_ImVol, RGBW_channels=prefs)

    nwbfile.add_acquisition(NP_image)

    io = NWBHDF5IO('/Users/danielysprague/foco_lab/data/final_nwb/HL'+worm+'.nwb', mode='w')
    io.write(nwbfile)
    io.close()

    return

if __name__ == '__main__':

    datapath = '/Users/danielysprague/foco_lab/data'

    
    strain_dict = {'20221028-18-48-00':'FC121', '20221106-21-00-09':'FC121', '20221106-21-23-19':'FC121',
                   '20221106-21-23-19':'FC121', '20221106-21-47-31':'FC121', '20221215-20-02-49':'FC121',
                   '20221215-22-02-55':'FC121', '20230412-20-15-17':'FC121', '20230322-18-57-04':'OH16230',
                   '20230322-20-16-50':'OH16230', '20230322-21-41-10':'FC128', '20230322-22-43-03':'FC128',
                   '20230506-12-56-00':'FC121', '20230506-13-32-08':'FC121', '20230506-14-24-57':'FC121',
                   '20230506-15-01-45':'OH16230', '20230506-15-33-51':'OH16230', '20230510-12-53-34':'FC121',
                   '20230510-13-25-46':'FC121', '20230510-15-49-47':'FC128', '20230510-16-36-46':'FC128',
                   '20230904-14-30-52':'FC111' , '20230904-15-09-05': 'FC111', '20230904-15-59-40': 'FC128',
                   '20230909-16-48-09':'FC121'}

    calc_datasets = ['20230322-18-57-04', '20230322-20-16-50', '20230506-15-33-51', '20230506-15-01-45']
    ignore_datasets = ['20230506-13-32-08', '20221215-22-02-55', '20230510-16-36-46', '20230909-14-26-56', '20230909-15-40-07']

    for folder in os.listdir(datapath+'/NP_Ray'):
        if folder == '.DS_Store' or folder in ignore_datasets:
            continue

        strain = strain_dict[folder]
        print(folder)
        t0 = time.time()
        if folder in calc_datasets:
            process_NP_FOCO_Ray(datapath, folder, strain, calc=True)
        else:
            process_NP_FOCO_Ray(datapath, folder, strain)
        t1 = time.time()
        print(t1-t0)
        break
    
    
    for folder in os.listdir(datapath+'/Yemini_21/OH16230/Heads'):
        if folder == '.DS_Store':
            continue
        print(folder)
        t0 = time.time()
        process_yemini(datapath + '/Yemini_21/OH16230/Heads/'+folder)
        t1 = time.time()
        print(t1-t0)


    for folder in os.listdir(datapath+'/NP_FOCO_cropped'):
        if folder == '.DS_Store':
            continue
        strain = 'OH16230'

        print(folder)

        if folder == '2022-04-26-w00-NP1' or folder == '2022-04-26-w01-NP1':
            process_NP_FOCO_original(datapath, folder, strain, calc=True)
        else:
            process_NP_FOCO_original(datapath, folder, strain, calc=False)

    
        
    for file in os.listdir(datapath+'/NP_paper/all'):
        if file[-4:] == '.mat' and file[-6:]!='ID.mat':
            print(file)
            process_Yem_original(file)


    for folder in os.listdir(datapath+ '/NP_chaudhary'):
        print(folder)
        if folder == '.DS_Store':
            continue
        process_chaudhary(datapath+ '/NP_chaudhary/'+ folder)
