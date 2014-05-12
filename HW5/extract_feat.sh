dir=~/HW5/demo_reco/*.wav
for f in $dir
do
	echo $f
	x=${f:20:20}.wav
	echo $x
	ch_wave $f -scaleN 0.65 -c 0 -F 16000 -o $x
	#rm -f $f
done
#rm -rf test_recordings/output.wav
#rm -rf *.mfcc
python extract_feat_dir.py demo_reco