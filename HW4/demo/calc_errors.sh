#!/usr/bin/bash
file_name='unsegmented.txt'
echo $file_name
output_file_name='result.txt'

python working.py $file_name $output_file_name
number_of_words=`cat $output_file_name | wc -w`
errors=`expr 161 - $number_of_words`

err=`echo $errors | awk '{ print ($1 >= 0) ? $1 : 0 - $1}'`

nerr=`python load_dict.py $output_file_name`
echo $nerr
echo $err

