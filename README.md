# NWBelegans: Tools for converting to and analyzing NWB files for _C. elegans_ whole-brain activity and NeuroPAL imaging
### Author: Daniel Sprague



## Installation instructions

1. Setup a python virtual environment and activate that environment (we typically use Anaconda)

2. Fork this repository to your local drive

3. Direct to the root of the local repository and run the command 'pip install .' from the command line 

4. You will also need to run 'python -m ipykernel install --user --name=*name of your environment*' to install the virtual environment as an ipython kernel



NWB is an HDF5 based standard data format for neuroscience data with APIs in both Python and Matlab. We think that standardizing to this common format will enable ease of collaboration and data sharing between labs.

This repository contains tutorials and examples of conversions of NeuroPAL structural images and whole-brain calcium imaging data and metadata to the standard NWB format as well as example code for various analyses that can be done using converted NWB files.  

Start with the NWB_tutorial.ipynb file which provides a walkthrough of the basics of NWB, creating objects, adding data, and writing
the NWB file to disk. Please read the NWB file components section of this README as well for descriptions of the contents of the NWB files and conventions for naming objects. Go to https://ucsf.box.com/s/8kbdfywefcfsn4pfextrzcr25az1vmuj for a video tutorial and a folder containing the example data used in the conversion tutorial and the various example analyses in this repository. Please download this data folder and add it to your local version of this repository.

After you have created your NWB files, follow the instructions at https://www.dandiarchive.org/handbook/13_upload/ to upload your data 
to Dandi.

This project is a work in progress and we hope to continue to develop it in collaboration with other *C. elegans* neuroscientists. Please
reach out to daniel.sprague@ucsf.edu if you have any questions about this work, are having issues with your own data conversion, or have 
suggestions for improvement.


# Analysis folder

This folder contains the code used for all of the analyses conducted in Sprague et al. 2024 [TODO: attach link to biorxiv preprint when available]. See the ReadMe within this folder for information about how to use this code. You can either stream existing NWB datasets from the dandiarchive or run the same analyses on your NWB datasets.

# NWB file components

Below are the main objects that should be used in an NWB file for saving NeuroPAL structural images and whole-brain calcium images. We ask that for objects listed here, you keep the object names and data formats consistent to the extent possible. NWB objects are indexed based on their names so consistency is important so that all NWB files from different groups can be easily input into the same processing pipeline. 

For types of data that you want to add to your NWB file but are not listed here, feel free to give objects names and descriptions that make most sense to you or reach out and suggest a standard that should be used for that type of data.

Object attributes will be listed below as 
'attribute name' - 'description of attribute' ('data type of attribute')

## NWBFile: 
This is the base file object which will contain metadata on the overall experiment.

### Attributes:
session_description - description of the experiment (text) <br>
identifier - identifier label for this experiment (text) <br>
session_start_time: time when the experiment started (datetime) <br>
lab - name of lab acquiring data (text) <br>
institution - name of institution where data was acquired (text) <br>
related_publications - publications associated with this dataset (text) <br>

## CElegansSubject:
This is the subject file which containes metadata about the subject of the experiment.

### Attributes:
subject_id - unique identifier for this animal (text) <br>
date_of_birth - date of birth of the animal (datetime) <br>
growth_stage - current growth stage of the animal (text) <br>
growth_stage_time - time spent in current growth stage (timedelta) <br>
cultivation_temp - temp animals were raised in (float) <br>
description - description of the subject (text) <br>
species - species of the animal (text) <br>
sex - sex of the animal, use O for hermaphrodite or M for male (text) <br>
strain - strain of the animal (text) <br>

## Device:
Device used to acquire data, can create multiple objects if multiple devices for 
different images

### Attributes: <br>
name - name of the object, should be unique for each device (text) <br>
description - free form description of device (text) <br>
manufacturer - manufacturer of the device (text) <br>

## NeuroPALImagingVolume:
Instance of an ImagingVolume object. Metadata associated with the acquisition of the NeuroPAL structural image. The name of this object should be 'NeuroPALImVol'.

### Attributes: <br>
name - 'NeuroPALImVol <br>
optical_channel_plus - instance of OpticalChannelPlus object (described below) <br>
order_optical_channels - instance of OpticalChannelRefs object (described below) <br>
description - free form description of ImagingVolume (text) <br>
device - device used for acquisition (NWB Device object described above) <br>
location - what part of the body of the worm is being imaged (text) <br>
grid_spacing - the voxel spacing or pixel resolution of the image in x, y, z (list of floats) <br>
grid_spacing_unit - the unit of grid_spacing, most likely micrometers (text)<br>
origin_coords - carry over from other model organisms, can just set as [0,0,0] (list of integers)<br>
origin_coords_unit - unit of origin_coods (text)<br>
reference_frame - carry over from other model organisms, can just set as 'worm head/tail' (text)

## OpticalChannelPlus
Contains metadata for each of the channels used in the image.

### Attributes: <br>
name - name of the channel, should be the fluorophore used (text) <br>
description - description of the channel, should be the filter used (text) <br>
excitation_lambda - the wavelength of the excitation laser nm (float) <br>
excitation_range - range of laser, should just be +-1.5nm around excitation_lambda (list of floats) <br>
emission_range - range of emission filter in nm (list of floats) <br>
emission_lambda - center of emission filter in nm (float) <br>

## OpticalChannelRefs
References to the OpticalChannelPlus of the OpticalChannelPlus objects which maintains the ordering of the channels in the image. The list of OpticalChannelPlus objects does not necessarily maintain ordering which is why we must use this object.

### Attributes: <br>
name - should be 'OpticalChannelRefs' for NeuroPAL image <br>
channels - ordered list of channel descriptors "excitation_lambda - emission_center - emission_range" (text) <br>

## NeuroPALImageRaw:
Instance of MultiChannelVolume containing the raw NeuroPAL image. Data will be saved and later loaded as a numpy aarray.

### Attributes: <br>
name - "NeuroPALImageRaw" (text) <br>
description - free form description of the image (text) <br>
RGBW_channels - which channels of the image correspond to the RGBW pseudocolors (list of integers) <br>
data - raw image data in dimension order [channels, Z, Y, X] (matrix of ints)
imaging_volume - link to ImagingVolume object defined above (ImagingVolume object)

## NeuroPALSegmentation:
Consists of two objects: one instance of an ImageSegmentation object and one of a PlaneSegmentation object. See names specified in attributes.

### ImageSegmentation Attributes: <br>
name - 'NeuroPALSegmentation' <br>
PlaneSegmentations - segmentations of the NeuroPAL image (one or more PlaneSegmentation objects)

### PlaneSegmentation Attributes: <br>
name - main segmentation should be called 'NeuroPALNeurons' (text) <br>
description - description of what is being segmented (text) <br>
imaging_plane - reference to the ImagingVolume ojbect descriped above (ImagingVolume object) <br>

You will then add either voxel_masks or image_masks to the PlaneSegmentation objects for the segmentation. Both work by adding rows to an HDF5 DynamicTable which the PlaneSegmentation is built off of. <br>

### Voxel_masks:
Each ROI is represented by a list of voxels that belong to that ROI. Each entry in this list will have [x,y,z,weight]. Weight value should be a float value which can have any meaning, make sure to specify the meaning of this value in the description of the segmentaiton. <br>

### Image_masks:
Each ROI represented by an array that is same size as the original image where non-ROI voxels have value 0 and ROI voxels have non-zero value. <br>

You can then optionally add a column to the DynamicTable specifying the labels for the ROIs. In our case this will be the cell ID labels. Labels should be a list with the same length as the number of ROIs, using an empty string where there are no labels.

You may include other segmentations here but make sure they are labeled clearly and that the primary segmentation are names as described here.

## CalciumImagingVolume
Instance of an ImagingVolume object. Metadata associated with the acquisition of the Calcium image series. The name of this object should be 'CalciumImVol'.

### Attributes: <br>
name - 'CalciumImVol' (text) <br>
optical_channel_plus - instance of OpticalChannelPlus object (described below) <br>
order_optical_channels - instance of OpticalChannelRefs object (described below) <br>
description - free form description of ImagingVolume (text) <br>
device - device used for acquisition (NWB Device object described above) <br>
location - what part of the body of the worm is being imaged (text) <br>
grid_spacing - the voxel spacing or pixel resolution of the image in x, y, z (list of floats) <br>
grid_spacing_unit - the unit of grid_spacing, most likely micrometers (text)<br>
origin_coords - carry over from other model organisms, can just set as [0,0,0] (list of integers)<br>
origin_coords_unit - unit of origin_coods (text)<br>
reference_frame - carry over from other model organisms, can just set as 'worm head/tail' (text) <br>

## CalciumImageSeries
Instance of MultiChannelVolumeSeries object. Data will be saved and later loaded as a numpy array. Name for the raw data should be 'CalciumImageSeries'.

### Attributes: <br>
name - 'CalciumImageSeries' (text) <br>
description - free form text description (text) <br>
comments - additional comments about this image series (text) <br>
data - raw image data in dimension order [time, X, Y, Z, channels] (array of ints) <br>
device - link to device object used to acquire this image (device object) <br>
unit - unit of the data input here (text) <br>
scan_line_rate - number of lines scanned per second, should be frame acquisition rate * numy planes * numz planes (float) <br>
dimension - size of each of the x, y, z dimensions (list of ints) <br>
resolution - smallest meaningful distance in specified unit between data values (float) <br>
rate - frame acquisition rate in hz (float) <br>
imaging_volume - link to ImagingVolume object described above 

## CalciumImageSegmentation
Will consist of an ImageSegmentation object and a series of PlaneSegmentation objects for each time point in the ImageSeries. 

The name of the ImageSegmentation algorithm should be 'CalciumSeriesSegmentation' if neurons are being tracked across frames (ie blob 1 in frame 0 corresponds to the same blob 1 in every other frame) or 'CalciumSeriesSegmentationUntracked' if every frame just contains the raw segmentation data for that frame itself (ie no correspondence between neurons across time points).

### ImageSegmentation attributes: <br>
name - should be 'CalciumSeriesSegmentation' or 'CalciumSeriesSegmentationUntracked' (text) <br>
plane_segmentations - list of PlaneSegmentation objects to add (list of PlaneSegmentation objects) <br>

### PlaneSegmentation attributes: <br>
name - should be 'Seg_tpoint_(t)' where '(t)' is the index of the current frame (text) <br>
description - free form description of this segmentation (text) <br>
imaging_plane - link to ImagingVolume object described above (ImagingVolume object) <br>
reference_images - link to image series this segmentation refers to (MultiChanelVolumeSeries object) <br>

Just as before, you can add ROIs either a voxel_mask or image_mask and labels by adding a column of labels to the PlaneSegmentation.

You may include other segmentations here but make sure they are labeled clearly and that the primary segmentation are names as described here.

## CalciumActivityData
We will save fluorescence data as DfOverF and/or Fluorescence objects depending on whether you have DfOverF values, raw fluorescence, or both. Start by creating an roi_table_region object on the first segmentation in the Calcium Time Series. Then you will load the data into an RoiResponseSeries object which will then be added to the DfOverF or Fluorescence object.

### RoiTableRegion attributes: <br>
description - free form description of what roi table region refers to, in this case probably 'all segmented neurons associated with calcium image series' (text) <br>
region - indices of ROIs in PlaneSegmentation to be included in this ROI region, probably just all ROIs for this purpose (list of ints) <br>

### RoiResponseSeries attributes: <br>
name - 'SignalCalciumImResponseSeries' (text) <br>
description - free form text description (text) <br>
data - activity data, first dimension should be time and second dimension should be ROIs (array of floats) <br>
rois - RoiTableRegion defined above (RoiTableRegion)

### DfOverF/Fluorescence attributes:
name - 'SignalDFoF' for DfOverF or 'SignalRawFluor' for raw fluorescence
roi_response_series - RoiResponseSeries defined above (RoiResponseSeries)

Some labs use both a signal GCaMP channel and a reference channel for motion correction and artifact removal. If your data is like this, please name the signal channel as provided above. Name the reference channel RoiResponseSeries as 'ReferenceCalciumImResponseSeries' and DfOverF/Fluorescence as 'ReferenceDFoF'/'ReferenceRawFluor'. 

After processing the activity data, name the processed RoiResponseSeries as 'ProcessedCalciumImResponseSeries' and DfOverF/Fluorescence as 'ProcessedDFoF'/'ProcessedRawFluor'





