audio_files=( 'ph_nos/0_1.mfcc' 'ph_nos/0_2.mfcc' 'ph_nos/0_3.mfcc' 'ph_nos/0_4.mfcc' 'ph_nos/0_5.mfcc' 'ph_nos/1_1.mfcc' 'ph_nos/1_2.mfcc' 'ph_nos/1_3.mfcc' 'ph_nos/1_4.mfcc' 'ph_nos/1_5.mfcc' 'ph_nos/2_1.mfcc' 'ph_nos/2_2.mfcc' 'ph_nos/2_3.mfcc' 'ph_nos/2_4.mfcc' 'ph_nos/2_5.mfcc' 'ph_nos/3_1.mfcc' 'ph_nos/3_2.mfcc' 'ph_nos/3_3.mfcc' 'ph_nos/3_4.mfcc' 'ph_nos/3_5.mfcc' 'ph_nos/4_1.mfcc' 'ph_nos/4_2.mfcc' 'ph_nos/4_3.mfcc' 'ph_nos/4_4.mfcc' 'ph_nos/4_5.mfcc' 'ph_nos/5_1.mfcc' 'ph_nos/5_2.mfcc' 'ph_nos/5_3.mfcc' 'ph_nos/5_4.mfcc' 'ph_nos/5_5.mfcc' )

fileno=( 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 )

for i in "${fileno[@]}"
do
    echo $i
    echo ${audio_files[$i]}
    flnam=${audio_files[$i]}
    echo $flnam
    #echo $i
    python HW5_mod.py $i $flnam
done
