#!/bin/bash

python test_rec.py

# Converting audio to 16KHz mono
ch_wave output.wav -scaleN 0.65 -c 0 -F 16000 -o right_output.wav
rm -f output.wav

# Replaying recorded audio
python test_play.py right_output.wav

# Extract feats from audio
python extract_feats.py right_output.wav

# Plot and display images
/Applications/Matlab/MATLAB_R2012a.app/bin/matlab -nodisplay -nosplash < plotter.m
open Mag_spec.eps &
open Mel_spec.eps &
open Mel_log_spec.eps &

