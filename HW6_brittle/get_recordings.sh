#!/bin/bash


DIR_NAME=$1
REC_SESS=$2

if [ ! -d $DIR_NAME ]
then
    mkdir $DIR_NAME
fi


inp=n

for i in {0..5}
do

    if [ $i == 0 ]
    then
	x="0 1 2 3 4 5 6 7 8 9"
    
    elif [ $i == 1  ]
    then
	x="9 8 7 6 5 4 3 2 1 0"

    elif [ $i == 2  ]
    then
	x="1 2 3 4 5 6 7 8 9 0"

    elif [ $i == 3  ]
    then
	x="0 9 8 7 6 5 4 3 2 1"

    elif [ $i == 4  ]
    then
	x="1 3 5 7 9 0 2 4 6 8"
	
    else
	x="8 6 4 2 0 9 7 5 3 1"
    fi
    
    while [ $inp == "n" ]
    do
	echo "Say $x"
	sleep 1
	python ../test_rec.py
	python ../test_play.py output.wav

	echo "Is Recording ok? y/n "
	read inp

    done

    inp=n
    ch_wave output.wav -scaleN 0.65 -c 0 -F 16000 -o $DIR_NAME/${i}_${REC_SESS}.wav

done

rm -f output.wav