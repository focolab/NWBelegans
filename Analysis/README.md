# Code for analyzing NWB files of _C. elegans_ whole-brain activity and NeuroPAL imaging
### Author: Daniel Sprague

This folder contains the code that was used to generate the figures in [Unifying community-wide whole-brain imaging datasets enables robust automated neuron identification and reveals determinants of neuron positioning in _C. elegans_](https://www.biorxiv.org/content/10.1101/2024.04.28.591397v1).

All necessary data for running the example code provided in this repository can be found on [box](https://ucsf.box.com/s/ofgt45dcc3zw093ppfop879gc4cqjf3t). You can download this whole folder and place it within the NWBelegans root directory. The larger NWB file datasets can either be downloaded or streamed directly from the [DANDI archive](https://dandiarchive.org). Further details and examples on how to stream this data can be found in Analysis/process_file.py.

Access the figure_\[1-5\].ipynb files to reproduce the plots for each figure in the paper. These notebooks should be setup to run as is but you may have to adjust some of the paths to the location that data is stored on your local computer. Processing NWB files will be much faster if they are downloaded onto your local drive, but if you stream from online you will not have to download these sometimes extremely large files. You can flexibly choose to load from local or online for each dataset. You can also flexibly substitute your own data into each of these analyses.
