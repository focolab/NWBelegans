# C. elegans optophysiology NWB conversion
## Author: FoCo Lab

This repository contains tutorials and examples of conversions of NeuroPAL structural images and whole-brain calcium imaging data 
and metadata to the standard NWB format. 

Start with the NWB_tutorial.ipynb file which provides a walkthrough of the basics of NWB, creating objects, adding data, and writing
the NWB file to disk.

Create_NWB.ipynb has additional examples of converting datasets from various file formats. We hope to continue to update this page 
with other examples of converting a range of different types of data to help researchers new to NWB convert their data quickly and 
efficiently.

After you have created your NWB files, follow the instructions at https://www.dandiarchive.org/handbook/13_upload/ to upload your data 
to Dandi.

This project is a work in progress and we hope to continue to develop it in collaboration with other *C. elegans* neuroscientists. Please
reach out to daniel.sprague@ucsf.edu if you have any questions about this work, are having issues with your own data conversion, or have 
suggestions for improvement.

## Installation instructions

1. Setup a python virtual environment and activate that environment

2. Fork this repository to your local drive

3. Direct to the root of the local repository and run the command 'pip install .' from the command line 

4. You will also need to run 'python -m ipykernel install --user --name=*name of your environment*' to install the virtual environment as an ipython kernel

