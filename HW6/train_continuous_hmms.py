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

        self.num_states = 5

        self.states=[]

        for i in xrange(1, self.num_states + 2): # The plus two is for non-emitting states
            # Non-emitting states first
            j=0
            if i == 1 :
                self.states.append(state(means=0, vars=0, self_trans=trans_prob[i][0],
                                   next_trans=trans_prob[i][1], 
                                   next_next_trans=trans_prob[i][2],non_emitting=True))
            else:
                self.states.append(state(means=emis_prob[j],
                                   vars=emis_prob[j+1],
                                   self_trans=trans_prob[i][0],
                                   next_trans=trans_prob[i][1],
                                   next_next_trans=trans_prob[i][2]))
                j = j + 2

        pass
            


def train_hmm(mapped_symbs, filenm):
    
    hmms=[]
    
    # Create HMM 
    for i in xrange(len(mapped_symbs)):
        hmms.append(hmm(symbol=mapped_symbs[i]))


    # Read in filenm
    data = np.loadtxt(filenm)


    # Create Matrix with means for cdist
    # Let's ignore the variances since they are all 1 anyway
    mean_matrix = np.zeros([len(hmms)*hmms[0].num_states,39])
    
    # prev-prev-pointer, prev-pointer
    # pointer -2 means prev is non-emitting state
    # pointer -1 means no parent
    parent_matrix = np.ones([len(hmms)*hmms[0].num_states,2])*-1

    # HMM/state matrix
    # First column specifies the HMM number. 
    # Second column refers to the state of that HMM
    # [4,1] means hmms[4].states[1]
    # This is just for rapidly indexing the HMM and state
    hmm_state_mat = np.ones([len(hmms)*hmms[0].num_states,2])*-1
    

    k = 0
    for i in xrange(len(hmms)):
        
        for j in xrange(1,hmms[i].num_states+1):
            
            mean_matrix[k,:] = hmms[i].states[j].means
            
            if j == 1:
                parent_matrix[k,:] = [-1, -2]
            elif j == 2:
                parent_matrix[k,:] = [-1, k-1]
            else:
                parent_matrix[k,:] = [k-2,k-1]
            
            hmm_state_mat[k,:] = [i,j]

            k = k + 1
            
    
    # Compute Euclidean distance between the means and the frames
    # of data

    DTW_dist = scipy.spatial.distance.cdist(mean_matrix,data,
                                            'mahalanobis',
                                            VI=np.eye(39))

    # Traverse DTW matrix and find shortest path -Anurag's code
    m,n = np.shape(DTW_dist)
    dcost = np.ones((m+2,n+1))
    dcost = dcost + np.inf

    DTW_bptr = np.zeros((m+2,n+1))
    DTW_bptr = DTW_bptr + np.inf
    

    dcost[2,1] = DTW_dist[0,0]

    k=3
    for j in range(2,n+1):
       for i in range(2,min(2+k,m+2)):

           prev_parent = parent_matrix[i-2][1]
           pp_parent = parent_matrix[i-2][0]

           if prev_parent == -2:
               # Parent is non-emitting state
               # Double check if this is sum or product
               costs = np.array([dcost[i,j-1]
                                 +hmms[int(hmm_state_mat[i-2][0])].states[int(hmm_state_mat[i-2][1])].self_trans,
                                 dcost[i-1,j-1]])
               
           elif pp_parent == -1:
               costs = np.array([dcost[i,j-1]
                                 + hmms[int(hmm_state_mat[i-2][0])].states[int(hmm_state_mat[i-2][1])].self_trans,
                                 dcost[i-1,j-1]
                                 + hmms[int(hmm_state_mat[i-3][0])].states[int(hmm_state_mat[i-3][1])].next_trans])
               
           else:
               costs = np.array([dcost[i,j-1]
                                 + hmms[int(hmm_state_mat[i-2][0])].states[int(hmm_state_mat[i-2][1])].self_trans,
                                 dcost[i-1,j-1]
                                 + hmms[int(hmm_state_mat[i-3][0])].states[int(hmm_state_mat[i-3][1])].next_trans,
                                 dcost[i-2,j-1]
                                 + hmms[int(hmm_state_mat[i-4][0])].states[int(hmm_state_mat[i-4][1])].next_next_trans ])
               

           
#            costs = np.array([dcost[i,j-1]+trans_mat[i][0], #same state
#                             dcost[i-1,j-1]+trans_mat[i-1][1],#prev state
#                            dcost[i-2,j-1]+trans_mat[i-2][2]])#prev-prev-state
           dcost[i,j] = np.min(costs) +DTW_dist[i-2,j-1]
           tmp_ptr = np.argmin(costs)

           if tmp_ptr == 0:
               DTW_bptr[i,j] = i
           elif tmp_ptr == 1:
               DTW_bptr[i,j] = prev_parent
               if prev_parent == -2:
                   DTW_bptr[i,j] = i-1
           else:
               DTW_bptr[i,j] = pp_parent
       k=k+2

    # Segmentations: (No of HMMS * number of states)-1 cuts
    segs = np.zeros([(len(hmms)*hmms[0].num_states)-1,1])
       
    # Backtracking
    j = n
    btrace = np.zeros((n+1,))
    trans_count = np.zeros((len(hmms)*hmms[0].num_states,
                            len(hmms)*hmms[0].num_states))

    prev = m + 1
    current = m + 1

    btrace[0] = 2
    btrace[1] = 2
    while j >= 2:
        btrace[j] = prev
        current = DTW_bptr[prev][j]
        j = j - 1
        trans_count[current-2,prev-2] = trans_count[current-2,prev-2] + 1
        prev = current
        
    btrace = btrace -2

    binct = np.bincount(btrace.astype(np.int64,casting='unsafe'))
    

    

    pass



def train_cont_hmms(transcrps, dirname):
    
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
        fullname = dirname + '/' + filenm + '.mfcc'
        # Put models together and train
        train_hmm(mapped_symbs, fullname)
        

    pass

    
    


if len(sys.argv) <=2:
    print "Usage:\npython train_continuous_hmms.py TRANSCRIPs DIR"
    print "Assumes that the models/ directory exists\n"
    exit(0)

if __name__ == '__main__':
    train_cont_hmms(sys.argv[1],sys.argv[2])
