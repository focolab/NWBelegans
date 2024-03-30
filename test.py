
from pynwb import NWBFile, NWBHDF5IO
from pynwb import register_class

from pynwb.file import Subject
from pynwb.ophys import PlaneSegmentation

from hdmf.utils import docval, popargs, get_docval

from ndx_multichannel_volume import CElegansSubject, OpticalChannelReferences, OpticalChannelPlus, ImagingVolume, VolumeSegmentation, MultiChannelVolume, PlaneExtension
from pynwb.ophys import OnePhotonSeries, OpticalChannel, ImageSegmentation, Fluorescence, CorrectedImageStack, MotionCorrection, RoiResponseSeries, PlaneSegmentation, ImagingPlane
import numpy as np

from datetime import datetime
from dateutil import tz


#Below is the code used to define the extended class

#the following code was used to create the minimum working example failing NWB file
nwbfile = NWBFile(
    session_description = 'Add a description for the experiment/session. Can be just long form text',
    #Can use any identity marker that is specific to an individual trial. We use date-time to specify trials
    identifier = '20230322-21-41-10',
    #Specify date and time of trial. Datetime entries are in order Year, Month, Day, Hour, Minute, Second. Not all entries are necessary
    session_start_time = datetime(2023, 3, 22, 21, 41, 10, tzinfo=tz.gettz("US/Pacific")),
    lab = 'FOCO lab',
    institution = 'UCSF',
    related_publications = ''
)

nwbfile.subject = Subject(
    subject_id="001",
    age="P90D",
    description="mouse 5",
    species="Mus musculus",
    sex="M",
)

device = nwbfile.create_device(
    name="Microscope",
    description="My two-photon microscope",
    manufacturer="The best microscope manufacturer",
)

optical_channel = OpticalChannel(
    name="OpticalChannel",
    description="an optical channel",
    emission_lambda=500.0,
)

imaging_plane = nwbfile.create_imaging_plane(
    name="ImagingPlane",
    optical_channel=optical_channel,
    imaging_rate=30.0,
    description="a very interesting part of the brain",
    device=device,
    excitation_lambda=600.0,
    indicator="GFP",
    location="V1",
    grid_spacing=[0.01, 0.01],
    grid_spacing_unit="meters",
    origin_coords=[1.0, 2.0, 3.0],
    origin_coords_unit="meters",
)

img_seg = ImageSegmentation(
    name = 'NeuroPALSegmentation'
)

ps = PlaneExtension(
    name = 'PlaneSeg',
    description = 'test plane extension',
    imaging_plane = imaging_plane,
)

for _ in range(30):
    # randomly generate example starting points for region
    x = np.random.randint(0, 95)
    y = np.random.randint(0, 95)
    z = np.random.randint(0, 95)

    # define an example 4 x 3 region of pixels of weight '1'
    pixel_mask = []
    for ix in range(x, x + 4):
        for iy in range(y, y + 3):
                pixel_mask.append([ix, iy, 1])

    # add pixel mask to plane segmentation
    ps.add_roi(pixel_mask=pixel_mask)

img_seg.add_plane_segmentation(ps)

ophys_module = nwbfile.create_processing_module(
    name="ophys", description="optical physiology processed data"
)

ophys_module.add(img_seg)

with NWBHDF5IO("/Users/danielysprague/foco_lab/data/Test/test.nwb","w") as io:
    io.write(nwbfile)
