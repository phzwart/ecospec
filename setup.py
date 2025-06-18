#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=7.0',
    'spectral>=0.23.1',
    'numpy>=1.19.0',
    'scikit-image>=0.18.0',
    'torch>=1.7.0',
    'einops>=0.3.0',
    'dlsia>=0.1.0',
    'zarr>=2.10.0',
]

test_requirements = [ ]

setup(
    author="Petrus H. Zwart",
    author_email='PHZwart@lbl.gov',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Python Boilerplate contains all the boilerplate you need to create a Python package.",
    entry_points={
        'console_scripts': [
            'ecospec=ecospec.cli:main',
        ],
    },
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='ecospec',
    name='ecospec',
    packages=find_packages(include=['ecospec', 'ecospec.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/phzwart/ecospec',
    version='0.1.0',
    zip_safe=False,
)
