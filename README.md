# P2P-ST

## Overview
This repository contains the core implementation of Scale-Transform Method of Medium-Low Pixel Level to Parcel-Level
Monitoring of Crop Land.

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
