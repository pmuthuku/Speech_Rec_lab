audio_files=( 'cont_recordings/0_1.mfcc' 'cont_recordings/0_2.mfcc' 'cont_recordings/0_3.mfcc' 'cont_recordings/0_4.mfcc' 'cont_recordings/0_5.mfcc' 'cont_recordings/1_1.mfcc' 'cont_recordings/1_2.mfcc' 'cont_recordings/1_3.mfcc' 'cont_recordings/1_4.mfcc' 'cont_recordings/1_5.mfcc' 'cont_recordings/2_1.mfcc' 'cont_recordings/2_2.mfcc' 'cont_recordings/2_3.mfcc' 'cont_recordings/2_4.mfcc' 'cont_recordings/2_5.mfcc' 'cont_recordings/3_1.mfcc' 'cont_recordings/3_3.mfcc' 'cont_recordings/3_4.mfcc' 'cont_recordings/3_5.mfcc' 'cont_recordings/4_1.mfcc' 'cont_recordings/4_3.mfcc' 'cont_recordings/4_4.mfcc' 'cont_recordings/4_5.mfcc' 'cont_recordings/5_1.mfcc' 'cont_recordings/5_2.mfcc' 'cont_recordings/5_3.mfcc' 'cont_recordings/5_4.mfcc' 'cont_recordings/5_5.mfcc' )

#fileno=( 0 4 5 9 10 14 15 18 19 22 23 27 )
fileno=( 1 2 3 6 7 8 11 12 13 16 17 20 21 24 25 26)
for i in "${fileno[@]}"
do
    echo $i
    echo ${audio_files[$i]}
    flnam=${audio_files[$i]}
    echo $flnam
    #echo $i
    python HW5_mod.py $i $flnam
done