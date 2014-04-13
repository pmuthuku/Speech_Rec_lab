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

    model_names = [ words2num[x] for x in words]
    
    return model_names, filenm

    
class state:

    def __init__(self, means, vars, self_trans, next_trans, next_next_trans,
                 non_emitting=False):
        '''means -> 39 dimensional mean
           vars -> 39 dimensional variances
           self_trans -> log probability of staying in same state
           next_trans, next_next_trans -> self explanatory'''
        self.means = means
        self.vars = vars
        self.self_trans = self_trans
        self.next_trans = next_trans
        self.next_next_trans = next_next_trans
        self.non_emitting = non_emitting



class hmm:
    def __init__(self, symbol):
        ''' symbol: the digit in question '''

        # Read in emission probabilities
        emis_prob = np.loadtxt('models/'+symbol+'.hmm')
        # Read in transition probabilities
        trans_prob = np.loadtxt('models/'+symbol+'.trans')

        num_states = 5

        self.states=[]

        for i in xrange(num_states + 2): # The plus two is for non-emitting states
            # Non-emitting states first
            j=0
            if i < 2:
                self.states.append(state(means=0, vars=0, self_trans=trans_prob[i][0],
                                   next_trans=trans_prob[i][1], 
                                   next_next_trans=trans_prob[i][2],non_emitting=True))
            else:
                self.states.append(state(means=emis_prob[j],
                                   vars=emis_prob[j+1],
                                   self_trans=trans_prob[i][0],
                                   next_trans=trans_prob[i][1],
                                   next_next_trans=trans_prob[i][2]))

        pass
            


def train_hmm(mapped_symbs, filenm):
    
    first_symb=mapped_symbs[0]
    hmm1 = hmm(symbol=first_symb)

    pass

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
        # Clean up names
        mapped_symbs,filenm = cleanup_names(trans[i])

        # Put models together and train
        train_hmm(mapped_symbs, filenm)
        

    pass

    
    


if len(sys.argv) <=2:
    print "Usage:\npython train_continuous_hmms.py TRANSCRIPs DIR"
    print "Assumes that the models/ directory exists\n"
    exit(0)

if __name__ == '__main__':
    train_cont_hmms(sys.argv[1],sys.argv[2])
