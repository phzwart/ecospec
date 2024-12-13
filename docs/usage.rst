=====
Usage
=====

To use ecoSpec in a project::

    import ecospec

=======
ecoSpec
=======

A Python toolkit for processing and analyzing hyperspectral imaging data from ecoFAB experiments. This package provides tools for:

* Automated alignment and registration of multi-view spectral data
* Spectral data normalization and preprocessing
* Deep learning-based segmentation of plant features
* Time series analysis of plant growth and health indicators

The package is designed to be modular and adaptable, allowing researchers to:
* Process standard ecoFAB experimental data with minimal setup
* Customize processing pipelines for specific experimental needs
* Extend the segmentation models for new plant features
* Integrate with existing analysis workflows

Example Usage
------------

Basic processing of ecoFAB data::

    import ecospec
    
    # Load and align multi-view data
    data = ecospec.process_view("path/to/data", view="front")
    
    # Normalize spectral data
    norm_data, bands = ecospec.normalize_data.prep_and_normalize(
        data.bands, 
        data.spectral_data
    )
    
    # Apply pre-trained segmentation model
    model = ecospec.load_pretrained_model("standard_segmentation")
    segments = model.predict(norm_data)

The package includes pre-trained models for common plant features, but can be adapted for specific needs through transfer learning or custom model architectures.

Installation
------------

Install via pip::

    pip install ecospec

For development installation::

    git clone https://github.com/phzwart/ecospec.git
    cd ecospec
    pip install -e .

Documentation
------------

Full documentation is available at https://ecospec.readthedocs.io/

License
-------
BSD License

Credits
-------
Developed at Lawrence Berkeley National Laboratory
