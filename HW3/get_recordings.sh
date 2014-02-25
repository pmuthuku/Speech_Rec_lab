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
    mv output.wav $DIR_NAME/$i.wav

done

