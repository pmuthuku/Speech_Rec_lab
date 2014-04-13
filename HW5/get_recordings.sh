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
    
    while [ $inp == "n" ]
    do
	
	num1=$RANDOM
	let "num1 %= 1000"
	num2=$RANDOM
	let "num2 %= 1000"
	num3=$RANDOM
	let "num3 %= 10000"

	printf "Say "
	
	printf "%03d " $num1
	printf "%03d " $num2
	printf "%04d\n" $num3
	
	sleep 1
	python ../test_rec.py
	python ../test_play.py output.wav

	echo "Is Recording ok? y/n "
	read inp

    done

    inp=n
    ch_wave output.wav -scaleN 0.65 -c 0 -F 16000 -o $DIR_NAME/${i}_${REC_SESS}.wav
    echo "$num1 $num2 $num3 (${i}_${REC_SESS})" >> $DIR_NAME/transcrp

done

rm -f output.wav