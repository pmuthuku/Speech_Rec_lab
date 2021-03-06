#!/bin/bash

inp=n

while [ $inp == "n" ]
do
    sleep 1
    python test_rec.py
    python ../test_play.py output.wav
    
    echo "Is Recording ok? y/n "
    read inp
	
done

ch_wave output.wav -scaleN 0.65 -c 0 -F 16000 -o right_output.wav
python extract_feats.py right_output.wav

python test_hmm_live.py 

rm -f output.wav right_output.wav