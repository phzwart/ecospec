"""
Core functionality for ecospec.
"""

from .spectral_data_object import MultimodalDataset
from .reader import get_data
from .process import process_view, construct_series, align, parse_file_name
from .normalize_data import prep_and_normalize
from .get_reference import mask, mean
from .spectral_segmenter import specseg, FCNetwork, FCNetwork1D, specseg_from_file, build_random_network

__all__ = [
    'MultimodalDataset',
    'get_data',
    'process_view',
    'construct_series',
    'align',
    'parse_file_name',
    'prep_and_normalize',
    'mask',
    'mean',
    'specseg',
    'FCNetwork',
    'FCNetwork1D',
    'specseg_from_file',
    'build_random_network'
] 