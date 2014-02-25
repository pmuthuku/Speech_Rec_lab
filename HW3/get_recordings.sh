#!/bin/bash


DIR_NAME=$1

if [ ! -d $DIR_NAME ]
then
    mkdir $DIR_NAME
fi


inp=n

for i in {0..9}
do
    while [ $inp == "n" ]
    do
	echo "Say $i"
	sleep 1
	python test_rec.py
	python ../test_play.py output.wav

	echo "Is Recording ok? y/n "
	read inp

    done

    inp=n
    ch_wave output.wav -scaleN 0.65 -c 0 -F 16000 -o $DIR_NAME/$i.wav

done