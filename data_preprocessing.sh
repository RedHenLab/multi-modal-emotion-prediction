#!/bin/bash

dir = /home/karolina/Documents/GSOC/data/IEMOCAP_full_release/Session*/sentences/wav/Ses*/
cd /home/karolina//Documents/GSOC/libraries/pyAudioAnalysis

for d in /home/karolina/Documents/GSOC/data/IEMOCAP_full_release/Session*/sentences/wav/Ses*/; do (python audioAnalysis.py featureExtractionDir -i "$d"  -mw 1.0 -ms 1.0 -sw 0.020 -ss 0.01); done
