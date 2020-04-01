# LagrangianPressureExtraction

The purpose of this repository is to publish some code I wrote in graduate school for post-processing experimental data. The code takes as input Lagrangian velocimetry measurements (i.e. from Particle Tracking Velocimetry, Shake The Box, or other measurement techniques that produce Lagrangian velocimetry data) and outputs the pressure field on the unstructured data points. The process is not completely automated, however. User effort is required to properly create / assign a set of the data points to act as boundary conditions. Additionally, if your domain contains a solid boundary then you will need to modify the tessellation function to not connect data points through that solid boundary. This repository is more of a collection of functions that could be used to conduct this analysis, presented along with a Jupyter notebook demonstrating a simple analysis on some synthetic Lagrangian data.

This repository is under the MIT license, and anyone is free to take the provided modules for the Taylor-Green vortex or for Lagrangian data processing and use them for any application, be it private, academic, or commercial. If the Lagrangian data processing code is used for an academic work that results in publication, please cite one of the following papers:

 - Neeteson, N. J., Bhattacharya, S., Rival, D., Michaelis, D., Schanz, D. and Schroeder, A., 2016, Pressure-Field Extraction from Lagrangian Flow Measurements: First Experiences Using 4D-PTV Data, Experiments in Fluids, 57:102. 
    - DOI: http://link.springer.com/article/10.1007/s00348-016-2170-4 
  
- Neeteson, N. and Rival, D., 2015, Pressure-Field Extraction on Unstructured Flow Data Using a Voronoi Tessellation-Based Networking Algorithm: A Proof-of-Principle Study, Experiments in Fluids, 56, 44. 
  - DOI: http://link.springer.com/article/10.1007%2Fs00348-015-1911-0

