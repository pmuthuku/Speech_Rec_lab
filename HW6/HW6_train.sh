# first arg name of file to read
# sec arg name of dir where mffcs are stored
while read line; do
    #echo $line
    python HW6_1_train.py "$line" $2
done < $1
