"""
Data normalization utilities for hyperspectral data processing.

This module provides functionality to preprocess and normalize hyperspectral data,
including wavelength band selection, NaN handling, and standardization of spectral
measurements.
"""

import numpy as np
import torch

def prep_and_normalize(bands, 
                       hs_data, 
                       eps=1e-3, 
                       lower_limit=495, 
                       upper_limit=684):
    """
    Prepare and normalize hyperspectral data within specified wavelength bounds.

    Args:
        bands (numpy.ndarray): Array of wavelength bands.
        hs_data (numpy.ndarray): Hyperspectral data array with shape (samples, bands, height, width).
        eps (float, optional): Small constant to prevent division by zero. Defaults to 1e-3.
        lower_limit (float, optional): Lower wavelength bound in nm. Defaults to 495.
        upper_limit (float, optional): Upper wavelength bound in nm. Defaults to 684.

    Returns:
        tuple:
            - torch.Tensor: Normalized hyperspectral data
            - numpy.ndarray: Selected wavelength bands

    Notes:
        The normalization process includes:
        1. Wavelength band selection
        2. Conversion to PyTorch tensor
        3. NaN replacement with zeros
        4. Mean subtraction and standard deviation normalization
    """
    sel = (bands > lower_limit) & (bands < upper_limit)
    new_bands = bands[sel]
    data = hs_data[:,sel,...]
    data = torch.Tensor(data)
    sel = torch.isnan(data)
    data[sel]=0.0
    m = torch.mean(data, dim=1).unsqueeze(1)
    data = data - m
    s = torch.std(data, dim=1).unsqueeze(1)
    data = data / (s+eps)
    data[sel]=0.0
    return data, new_bands
