"""
Data reader module for spectral imaging data.

This module provides functionality to read and load both RGB images and 
hyperspectral data from ENVI format files. It handles data loading, basic
error handling, and necessary array transformations for consistent data
organization.

The organization of this code is very ecoBOT specific, as it relies on 
naming conventions implicitly defined in the ecoBOT protocols.
"""

import spectral.io.envi as envi
import os
from skimage.io import imread
import numpy as np
import einops

from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter


def get_data(path):
    """
    Load RGB and hyperspectral data from a specified directory.

    This function searches for and loads both RGB images (PNG format) and 
    hyperspectral data (ENVI format) from the 'results' subdirectory of the
    specified path. It specifically looks for files containing "REFLECTANCE"
    in their names.

    Args:
        path (str): Base path to the directory containing a 'results' subdirectory
            with the data files.

    Returns:
        tuple:
            - numpy.ndarray or None: RGB image data if found, None otherwise
            - numpy.ndarray or None: Hyperspectral data if found, None otherwise
            - numpy.ndarray or None: Wavelength bands if found, None otherwise

    Notes:
        - The hyperspectral data is rearranged from (X, Y, C) to (C, Y, X) format
        - The X dimension is flipped in the final output
        - Errors during ENVI file loading are caught and printed
    """
    is_ok = False
    img = None
    spectral = None
    bands = None
    for file in os.listdir(path+'results/'):
        if file[-4:]==".png":
            if "REFLECTANCE" in file:
                img = imread(path+"results/"+file)
        if file[-4:]==".hdr":
            if "REFLECTANCE" in file:
                hdr_path= path+"results/"+file
                try:
                    spectral = envi.open(hdr_path).load()
                    bands = spectral.bands.centers
                    spectral = np.array(spectral)
                    spectral = einops.rearrange(spectral, "X Y C -> C Y X")
                    spectral = spectral[:,:,::-1]
                    is_ok = True
                except:
                    print("There was an error with \n %s"%(hdr_path))
                    pass

    return img, spectral, bands
