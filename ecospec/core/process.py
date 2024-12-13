"""
Core processing module for ecospec data analysis.

This module provides functionality for aligning, processing, and organizing
spectral imaging data from multiple views of specimens. It includes tools for
image alignment using FFT-based methods, data processing pipelines, and 
time series construction.
"""

from ecospec.core import get_reference
from ecospec.core import reader
from ecospec.core import spectral_data_object
from skimage.filters import threshold_otsu
import numpy as np
import os
import einops


def align(A, ref):
    """
    Align two images using FFT-based cross-correlation.

    Args:
        A (numpy.ndarray): Source image to be aligned.
        ref (numpy.ndarray): Reference image to align against.

    Returns:
        tuple: A pair of integers (-ii, -jj) representing the vertical and 
            horizontal shifts needed to align the images.
    """
    ftA = np.fft.fft2(A)
    ftB = np.fft.fft2(ref)
    peak_map = np.fft.ifft2(ftA*ftB.conj()).real
    iijj = np.argmax(peak_map)
    ii,jj = np.unravel_index(iijj, A.shape)
    if ii > A.shape[0]//2:
        ii = ii - A.shape[0]
    if jj > A.shape[1]//2:
        jj = jj - A.shape[1]

    return -ii,-jj

def process_view(path_to_data, view):
    """
    Process a single view of spectral data, including alignment with a reference.

    Args:
        path_to_data (str): Path to the directory containing the spectral data.
        view (str): View identifier ("front", "side", or "top").

    Returns:
        tuple:
            - numpy.ndarray: Processed RGB image
            - numpy.ndarray: Processed spectral data
            - numpy.ndarray: Wavelength bands
    """
    ref_view = get_reference.mean(view)
    img, spec, bands = reader.get_data(path_to_data)
    tmp = np.sum(img.astype(float), axis=-1)
    otsu_img_threshold = threshold_otsu(tmp)
    sel = tmp > otsu_img_threshold

    tmp[sel] = 1.00
    tmp[~sel] = 0.00

    dii, djj = align(tmp, ref_view)

    img = np.roll(img, dii, axis=0)
    img = np.roll(img, djj, axis=1)

    spec = np.roll(spec, dii, axis=1)
    spec = np.roll(spec, djj, axis=2)

    return img, spec, bands

def parse_file_name(file):
    """
    Parse metadata from a standardized filename.

    Args:
        file (str): Filename to parse.

    Returns:
        dict: Dictionary containing parsed metadata with keys:
            - CODE: Experiment code
            - ID: Sample identifier
            - date: Acquisition date
            - view: View perspective (lowercase)
    """
    keys = file.split("_")
    meta = {"CODE":keys[0],
            "ID":keys[1],
            "date":keys[2],
            "view":keys[-1].lower()}
    return meta

def construct_series(paths_to_data, sample_id, zarr_file_name):
    """
    Construct a time series dataset from multiple spectral measurements.

    Args:
        paths_to_data (list): List of paths containing the raw data files.
        sample_id (str): Identifier for the sample being processed.
        zarr_file_name (str): Base name for the output Zarr store.

    Returns:
        list: List of MultimodalDataset objects, one for each view perspective.

    Notes:
        The function processes all available views ("front", "side", "top") and
        organizes the data chronologically. Each view's data is stored in a
        separate Zarr store with the naming pattern: {zarr_file_name}_{view}.zarr
    """
    sample_file_list = []
    meta_info = []
    for directory in paths_to_data:
        for file in os.listdir(directory):
            if sample_id in file:
                sample_file_list.append(directory+"/"+file)
                meta_info.append(parse_file_name(file))
    print(sample_file_list)

    views = ["front", "side", "top"]
    zarrs = []
    for view in views:
        imgs = []
        hss = []
        bandss = None
        time_points = []
        for file, meta in zip(sample_file_list, meta_info):
            if meta["view"] == view:
                img, spec, bands = process_view(file+"/", view)
                imgs.append(img)
                hss.append(spec)
                bandss = bands
                time_points.append(meta['date'])

        zarr_name = zarr_file_name+"_"+view+".zarr"
        imgs = einops.rearrange(imgs, "N Y X C -> N Y X C")
        hss = einops.rearrange(hss, "N C Y X -> N C Y X")
        time_points = np.array(time_points).astype(int)

        order = np.argsort(time_points)
        imgs = imgs[order,...]
        hss = hss[order,...]
        time_points = time_points[order]

        zarr_obj = spectral_data_object.MultimodalDataset(zarr_path=zarr_name,
                                                          sample_identifier=sample_id,
                                                          additional_metadata=view,
                                                          time_points=time_points,
                                                          hs_data=hss,
                                                          rgb_data=imgs,
                                                          bands=bands)

        zarrs.append(zarr_obj)
    return zarrs

















