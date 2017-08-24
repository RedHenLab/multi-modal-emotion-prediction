#!/bin/bash

# This is a preprocessing script for extracting audio features from audio files.
# This script runs over all relevant audio files from IEMOCAP data set and saves 
# numpy files with features in the folder containing the corresponding audio file.
# Make sure to install pyAudioAnalysis before the usage.
# Change paths in order to use it on your computer / with a different data set.


# Set the path to the data
dir = /home/karolina/Documents/GSOC/data/IEMOCAP_full_release/Session*/sentences/wav/Ses*/
# Enter the catalog with pyAudioAnalysis library tu run the python script
cd /home/karolina/Documents/GSOC/libraries/pyAudioAnalysis

# Extract short term features with window of 20 ms and overlap of 10 ms 
# and mid term features (not used for the project) with 1 s window.
for d in /home/karolina/Documents/GSOC/data/IEMOCAP_full_release/Session*/sentences/wav/Ses*/; do (python audioAnalysis.py featureExtractionDir -i "$d"  -mw 1.0 -ms 1.0 -sw 0.02 -ss 0.01); done
