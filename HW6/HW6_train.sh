# first arg name of file to read
# sec arg name of dir where mffcs are stored
for i in `seq 1 2`;
do
    while read line; do
	echo $line
	python HW6_1_train.py "$line" $2 $3
	#python parameter update code
    
    done < $1
    echo '*********************************************************'
done