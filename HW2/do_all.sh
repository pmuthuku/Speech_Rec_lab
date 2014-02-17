#!/bin/bash

# Do spell check on story
python HW2_f.py -f story.txt -d dict.txt -p n

# Create output in right format
perl format_story.pl Checked.txt > ourstory.txt

# Change originals to right format
perl format_story.pl story.txt > story_wpl.txt
perl format_story.pl storycorrect.txt > storycorrect_wpl.txt

# Errors in original
paste storycorrect_wpl.txt story_wpl.txt > story_storycorrect_wpl.txt
echo "Errors in original"
python word_by_word.py story_storycorrect_wpl.txt
echo " "
echo " "
# Errors in our story
paste storycorrect_wpl.txt ourstory.txt > story_ourstory_wpl.txt
echo "Errors in our story"
python word_by_word.py story_ourstory_wpl.txt

