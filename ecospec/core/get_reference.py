"""
Reference data loading utilities for the ecospec package.

This module provides functions to load pre-computed reference data (masks and mean images)
for different view perspectives of the specimen. The reference data is stored as NumPy
arrays in the package's reference directory.

Available views are:
    - "front": Front view of the specimen
    - "side": Side view of the specimen
    - "top": Top view of the specimen
"""

import ecospec
import numpy as np


data_path = ecospec.__path__[0]+"/reference/"
views = ["front", "side", "top"]

def mask(view):
    """
    Load and return the mask for a specific view.

    Args:
        view (str): The view perspective to load. Must be one of ["front", "side", "top"].

    Returns:
        numpy.ndarray: The binary mask array for the specified view.

    Raises:
        AssertionError: If the view is not one of the valid options.
    """
    assert view in views
    this_mask = np.load(data_path+view+"_mask.npy")
    return this_mask

def mean(view):
    """
    Load and return the mean reference image for a specific view.

    Args:
        view (str): The view perspective to load. Must be one of ["front", "side", "top"].

    Returns:
        numpy.ndarray: The mean reference image array for the specified view.

    Raises:
        AssertionError: If the view is not one of the valid options.
    """
    assert view in views
    this_mean = np.load(data_path + view + "_mean.npy")
    return this_mean





