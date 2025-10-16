# P2P-ST

## Overview
This repository contains the core implementation of **P2P-ST** for parcel-level remote sensing analysis. 
It provides Python scripts for high-resolution crop fraction estimation and zonal statistics calculation.

**Note:** Before applying the P2P-ST code, input images **must** be preprocessed to:
- unify spatial resolution,
- align grids,
- standardize coordinate systems (projection).

## Dependencies
- Python 3.8+
- numpy
- pandas
- arcpy (ArcGIS 10.8+)
- .etc

## Features
- **scale_transform.ipynb** – Implements pixel-to-parcel scale transformation
- **to_ENVI.py** – Handles ENVI format data management
- **unmixing.py** – Implements the unmixing procedure
- **unmixing_main.py** – Main script for executing the unmixing workflow
