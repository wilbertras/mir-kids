# Introduction
Welcome! This repository contains the code necessary to reproduce the figures of the paper: 

"Demonstrating Single-Photon Counting With Kinetic Inductance Detectors From 4 to 25 um"


This reproduction package has been made by Wilbert Ras-Vinke. 
For any questions please contact the author at w.ras@sron.nl


# Instructions
The raw data is provided seperately with its own DOI given in the paper. 
The data should be downloaded and the path to the data should be put into path2data.txt
Then the following scripts can be run. 

0. path2data.txt: make sure the path to the raw data is inserted here
1. initiate_kid.py: creates the subdirectories and fits Q-factors
2. initiate_filterstacks.py: generates and plots the filters (Fig. 7) and plots the radiances (Fig. 2a)
3. analyse_noise_spectra.py: plots the noise power spectral densities (Fig. 2b)
4. analyse_time_domain.py: plots the time domain pulse data (Fig. 3)
5. analyse_resolving_powers.ipynb: analyses the resolving powers (Fig. 5a) and plot the pulse height distributions (Fig. 4)
6. analyse_dark_count_rates.ipynb: analyses the coincident events (Fig. 8b) and finds the dark count rates based on the resolving powers (Fig. 5b and 9)
7. analyse_efficiency.py: fits the efficiency (Fig. 6)

functions.py: general functions
functions_meta.py: specific functions that combine a number of general functions

matplotlibcolors.py: defines the colors for plotting
matplotlibrc: style guide for plotting

S21/Khalil.py
S21/KID_S21.py









