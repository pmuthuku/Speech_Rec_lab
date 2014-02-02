#!/bin/bash

python test_rec.py

# Converting audio to 16KHz mono
ch_wave output.wav -scaleN 0.65 -c 0 -F 16000 -o right_output.wav
rm -f output.wav

# Replaying recorded audio
python test_play.py right_output.wav
