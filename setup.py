import setuptools
import os, platform


setuptools.setup(
    name="C. elegans optophys NWB",
    version="0.0.3",
    author="Daniel Sprague",
    author_email="daniel.sprague@ucsf.edu",
    description="Tutorials and guidelines for reading and writing NWB files for C. elegans optophysiology",
    long_description_content_type=open('README.md').read(),
    url="",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Framework :: napari",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy>=1.22.4',
        'scipy>=1.0.0',
        'tifffile>=2022.5.4',
        'opencv-python-headless>=4.1.0.25',
        'matplotlib>=2.1.0',
        'ndx-multichannel-volume',
        'pandas>=1.4.2',
        'ipython',
        'ipykernel',
        'scikit-image',
        'pynwb',
        'nwbinspector',
        'dandi',
        'networkx',
        'networkx',
        'adjustText',
        'latex',
        'PyWavelets',
        'seaborn',
        'scikit-learn',
        'remfile',
        'pyqt6'
    ], 
)
