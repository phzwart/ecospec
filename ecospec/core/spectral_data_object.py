"""
Multimodal dataset handling module using Zarr storage.

This module provides a class for managing multimodal spectral imaging data,
including hyperspectral and RGB data, using Zarr as a storage backend. It
handles data organization, storage, and retrieval with efficient chunked
array access.
"""

import zarr

class MultimodalDataset:
    """
    A class for handling multimodal data using Zarr as a storage backend.

    This class provides an interface for storing and retrieving multimodal
    imaging data, including hyperspectral data, RGB images, and associated
    metadata using Zarr arrays for efficient storage and access.

    Attributes:
        sample_identifier (str): Unique identifier for the sample.
        additional_metadata (str): Additional metadata for the sample.
        zarr_group (zarr.hierarchy.Group): Zarr group containing the stored data.
    """

    def __init__(self,
                 zarr_path,
                 sample_identifier=None,
                 additional_metadata=None,
                 time_points=None,
                 hs_data=None,
                 rgb_data=None,
                 bands=None):
        """
        Initialize a multimodal dataset with optional data.

        Args:
            zarr_path (str): Path where the Zarr file will be stored.
            sample_identifier (str, optional): Unique identifier for the sample.
            additional_metadata (str, optional): Additional sample metadata.
            time_points (numpy.ndarray, optional): Array of time points.
            hs_data (numpy.ndarray, optional): Hyperspectral data array.
            rgb_data (numpy.ndarray, optional): RGB image data array.
            bands (numpy.ndarray, optional): Wavelength bands information.

        Notes:
            Data arrays are stored in chunks for efficient access:
            - time_points: (1000,)
            - hs_data: (10, 10, 512, 512)
            - rgb_data: (10, 512, 512, 4)
            - bands: (1000,)
        """
        self.sample_identifier = sample_identifier
        self.additional_metadata = additional_metadata

        # Open a Zarr group on disk
        self.zarr_group = zarr.open_group(zarr_path, mode='a')

        # If data is provided, add it to the group
        if time_points is not None:
            self.zarr_group.create_dataset('time_points', data=time_points, chunks=(1000,), dtype='f8', overwrite=True)
        if hs_data is not None:
            self.zarr_group.create_dataset('hs_data', data=hs_data, chunks=(10, 10, 512, 512), dtype='f8', overwrite=True)
        if rgb_data is not None:
            self.zarr_group.create_dataset('rgb_data', data=rgb_data, chunks=(10, 512, 512, 4), dtype='i8', overwrite=True)
        if bands is not None:
            self.zarr_group.create_dataset('bands', data=bands, chunks=(1000,), dtype='f8', overwrite=True)

        # Add the sample identifier and additional metadata as Zarr attributes
        if sample_identifier is not None:
            self.zarr_group.attrs['sample_identifier'] = sample_identifier
        if additional_metadata is not None:
            self.zarr_group.attrs['additional_metadata'] = additional_metadata

    def get_hs_data(self):
        """
        Retrieve the stored hyperspectral data.

        Returns:
            numpy.ndarray: The complete hyperspectral data array.
        """
        return self.zarr_group['hs_data'][:]

    def get_rgb_data(self):
        """
        Retrieve the stored RGB image data.

        Returns:
            numpy.ndarray: The complete RGB image data array.
        """
        return self.zarr_group['rgb_data'][:]

    def get_bands(self):
        """
        Retrieve the wavelength bands information.

        Returns:
            numpy.ndarray: Array containing wavelength band centers.
        """
        return self.zarr_group['bands'][:]
