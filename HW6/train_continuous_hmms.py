import sys
import numpy as np
import scipy.spatial.distance


words2num = {'zero':'0',
             'oh' : 'o',
             'one': '1',
             'two': '2',
             'three' : '3',
             'four' : '4',
             'five' : '5',
             'six' : '6',
             'seven': '7',
             'eight' : '8',
             'nine' : '9',
             'sil ' : 'sil'}


def cleanup_names(seq_nums):
        
    words = seq_nums[:-1]
    filenm = seq_nums[-1]

    


def train_cont_hmms(transcrps, dirname):
    
    print "transcrps ",transcrps
    print "dirname ",dirname

    
    with open(transcrps) as f:
        trans = f.readlines()
        
    # Chomp trailing newline characters
    for i in xrange(len(trans)):
        trans[i] = trans[i].rstrip()
        trans[i] = trans[i].split()
        trans[i][-1] = trans[i][-1].replace('(','').replace(')','')


    # Read each line of the transcript, compose model and train
    for i in xrange(len(trans)):
        mapped_symbs = cleanup_names(trans[i])

    pass

    
    


if len(sys.argv) <=2:
    print "Usage:\npython train_continuous_hmms.py TRANSCRIPs DIR"
    print "Assumes that the models/ directory exists\n"
    exit(0)

if __name__ == '__main__':
    train_cont_hmms(sys.argv[1],sys.argv[2])
