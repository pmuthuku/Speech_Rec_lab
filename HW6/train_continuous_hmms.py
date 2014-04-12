import sys
import numpy as np
import scipy.spatial.distance


def train_cont_hmms(transcrps, dirname):
    
    print "transcrps ",transcrps
    print "dirname ",dirname

    
    with open(transcrps) as f:
        trans = f.readlines()
        
    # Chomp trailing newline characters
    for i in xrange(len(trans)):
        trans[i] = trans[i].rstrip()
        trans[i] = trans[i].split()
        trans[i][-1] = trans[i][-1]#######

    
    


if len(sys.argv) <=2:
    print "Usage:\npython train_continuous_hmms.py TRANSCRIPs DIR"
    print "Assumes that the models/ directory exists\n"
    exit(0)

if __name__ == '__main__':
    train_cont_hmms(sys.argv[1],sys.argv[2])
