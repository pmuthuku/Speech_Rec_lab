import sys
import numpy as np
import scipy.spatial.distance


DIRNAME = 'models_a_new/'
MODEL_DIR_NAME = 'models_a'
NO_OF_LEVELS = 10
NO_OF_HMM = 10 #only one hmm per level
NO_OF_STATES = 5

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

    
# class state:

#     def __init__(self, means, vars, self_trans, next_trans, next_next_trans,
#                  non_emitting=False):
#         '''means -> 39 dimensional mean
#            vars -> 39 dimensional variances
#            self_trans -> log probability of staying in same state
#            next_trans, next_next_trans -> self explanatory'''
#         self.means = means
#         self.vars = vars
#         self.self_trans = self_trans
#         self.next_trans = next_trans
#         self.next_next_trans = next_next_trans
#         self.non_emitting = non_emitting



# class hmm:
#     def __init__(self, symbol):
#         ''' symbol: the digit in question '''

#         # Read in emission probabilities
#         emis_prob = np.loadtxt('models/'+symbol+'.hmm')
#         # Read in transition probabilities
#         trans_prob = np.loadtxt('models/'+symbol+'.trans')

#         self.num_states = 5

#         self.states=[]

#         for i in xrange(1, self.num_states + 2): # The plus two is for non-emitting states
#             # Non-emitting states first
#             j=0
#             if i == 1 :
#                 self.states.append(state(means=0, vars=0, self_trans=trans_prob[i][0],
#                                    next_trans=trans_prob[i][1], 
#                                    next_next_trans=trans_prob[i][2],non_emitting=True))
#             else:
#                 self.states.append(state(means=emis_prob[j],
#                                    vars=emis_prob[j+1],
#                                    self_trans=trans_prob[i][0],
#                                    next_trans=trans_prob[i][1],
#                                    next_next_trans=trans_prob[i][2]))
#                 j = j + 2


#         pass

trans_names = [MODEL_DIR_NAME + '/0.trans',
               MODEL_DIR_NAME + '/1.trans',
               MODEL_DIR_NAME + '/2.trans',
               MODEL_DIR_NAME + '/3.trans',
               MODEL_DIR_NAME + '/4.trans',
               MODEL_DIR_NAME + '/5.trans',
               MODEL_DIR_NAME + '/6.trans',
               MODEL_DIR_NAME + '/7.trans',
               MODEL_DIR_NAME + '/8.trans',
               MODEL_DIR_NAME + '/9.trans'#,
               #MODEL_DIR_NAME + '/sil.trans'
];
hmm_names = [MODEL_DIR_NAME + '/0.hmm',
             MODEL_DIR_NAME + '/1.hmm',
             MODEL_DIR_NAME + '/2.hmm',
             MODEL_DIR_NAME + '/3.hmm',
             MODEL_DIR_NAME + '/4.hmm',
             MODEL_DIR_NAME + '/5.hmm',
             MODEL_DIR_NAME + '/6.hmm',
             MODEL_DIR_NAME + '/7.hmm',
             MODEL_DIR_NAME + '/8.hmm',
             MODEL_DIR_NAME + '/9.hmm'#,
             #MODEL_DIR_NAME + '/sil.hmm'
];

mu_list = [np.zeros((NO_OF_STATES, 39)) for x in range(10)]# no of hmm has beeb set as 1
sigma_list = [np.zeros((NO_OF_STATES, 39)) for x in range(10)]

trans_list = []

for hmm_idx, hmm_file in enumerate(hmm_names):
    f = np.loadtxt(hmm_file)
    for i in range(NO_OF_STATES):
        mu_list[hmm_idx][i] = f[i * 2]
        sigma_list[hmm_idx][i] = f[i * 2 + 1]

for trns_idx, trans_file in enumerate(trans_names):
    f = np.loadtxt(trans_file)
    trans = np.empty((NO_OF_STATES, NO_OF_STATES))
    trans.fill(np.inf) # comment out for +log case
    for j in range(NO_OF_STATES):
        if j > 2:
            subst = f[2 + j, 0:5 - j]
        else:
            subst = f[2 + j, :]

        trans[j, j:min(j + 3, NO_OF_STATES)] = subst#np.delete(f[2+j, :], np.where(f[2+j, :] == np.inf), axis=0)
    trans_list[len(trans_list):] = [trans]


class graph:
    template_nodes = []

    def add_node(self, template_node):
        if (len(self.template_nodes) <= template_node.identifier[3]):
            self.template_nodes.append([])
        self.template_nodes[template_node.identifier[3]].append(template_node)

    def find_node_by(self, level_no, hmm_no, state_no, time_seq=-1):
        this_time = self.template_nodes[time_seq]

        this_level = list(filter(lambda _: _.identifier[0] in level_no, this_time))
        this_hmm = list(filter(lambda _: _.identifier[1] in hmm_no, this_level))
        this_state = list(filter(lambda _: _.identifier[2] in state_no, this_hmm))
        return this_state

def calculate_C(xn, mu, sigma):
    inv_cov = np.linalg.inv(np.diagflat(sigma))
    tmp_dist = scipy.spatial.distance.cdist(np.matrix(mu), np.matrix(xn),
                                            #          'euclidean')
                                            'mahalanobis', VI=inv_cov)
    C = (0.5*tmp_dist)+(0.5*np.log(np.prod(sigma)))+(19.5*np.log(2*np.pi))
    return C


def calculate_P(parents, identifier, C):
    if len(parents) == 0:
        return (C, None)

    my_hmm_no = identifier[1]
    my_state_no = identifier[2]

    best_P_prev = np.inf
    path_idx = -1
    
    for par_idx, parent in enumerate(parents):
        parent_state_no = parent.identifier[2]
        parent_hmm_no = parent.identifier[1]
        if my_hmm_no == -1:
            trans_cost = 0
        # elif my_state_no == 0 and my_hmm_no == 10:
        #     trans_cost = -np.log(0.5)
        elif parent_hmm_no == -1:
            trans_cost = 0#-np.log(1)
        else :
            trans_cost = trans_list[my_hmm_no][parent_state_no, my_state_no]

        tr_cost = parent.P + trans_cost
        # print "Parent {0}, {1}-->tr_cost:{2}, parent_P:{3}".format(par_idx, parent.identifier, tr_cost, parent.P)
        if tr_cost < best_P_prev:
            best_P_prev = tr_cost
            path_idx = par_idx
    # print "\n"

    return (best_P_prev + C, parents[path_idx])

class template_node:
    def __init__(self, level_no, HMM_no, state_no, seq, time=-1, non_emitting=False):
        if (time == 0):
            self.parents = []
        elif (level_no == 0 and non_emitting == True):
            self.parents = []
        elif (non_emitting == True):
            self.parents = t_graph.find_node_by([level_no - 1], range(NO_OF_HMM), [NO_OF_STATES - 1], time_seq=time)#no of hmm has to be one
        elif (state_no == 0):
            self.parents = t_graph.find_node_by([level_no], [-1, HMM_no], [0], time_seq=time - 1)
        elif (state_no == 1):
            self.parents = t_graph.find_node_by([level_no], [HMM_no], range(state_no - 1, state_no + 1),
                                                time_seq=time - 1)
        elif (state_no > 1):
            self.parents = t_graph.find_node_by([level_no], [HMM_no], range(state_no - 2, state_no + 1),
                                                time_seq=time - 1)
        if (HMM_no != -1):
            self.mu = mu_list[HMM_no][state_no]
            self.sigma = sigma_list[HMM_no][state_no]
        self.non_emitting = non_emitting
        self.identifier = [level_no, HMM_no, state_no, time]

        # Compute C
        if(HMM_no != -1):
            if time==0:
                if state_no==0 and level_no==0:
                    C = calculate_C(seq[time,:], self.mu, self.sigma)
                else:
                    #C=-1*np.inf
                    C=np.inf
            else:
                C = calculate_C(seq[time,:], self.mu, self.sigma)
        else:
            if level_no == 0: #for third problem make it level_no ==0 and time==0
                #C = -1*np.inf
                C = np.inf
            elif time == 0:
                #C = -1*np.inf
                C = np.inf
            else:
                C = 0
        self.C = C
        (self.P, self.best_parent) = calculate_P(self.parents, self.identifier, self.C)

t_graph = None

def do_DTW(HMM, trans_mat, data):
    means = HMM[::2,:]
    varia = HMM[1::2,:]

    #varia = varia+ 0.001

    DTW_dist = np.zeros((5,data.shape[0]))

    for i in xrange(5):
        inv_cov = np.linalg.inv(np.diagflat(varia[i][:]))
        tmp_dist = scipy.spatial.distance.cdist(np.matrix(means[i][:]),data,
                                      #          'euclidean')
                                                'mahalanobis',VI=inv_cov)
        DTW_dist[i][:] = 0.5*tmp_dist+0.5*np.log(np.prod(varia[i][:]))+19.5*np.log(2*np.pi)

    np.savetxt('dist_file',DTW_dist)


    # Do actual DTW: Anurag's code
    m,n = np.shape(DTW_dist)
    #print m
    #print n
    dcost = np.ones((m+2,n+1))
    dcost = dcost + np.inf

    DTW_bptr = np.zeros((m+2,n+1))
    DTW_bptr = DTW_bptr + np.inf
    
    dcost[2,1] = DTW_dist[0,0]

    k=3
    for j in range(2,n+1):
       for i in range(2,min(2+k,m+2)):
           costs = np.array([dcost[i,j-1]+trans_mat[i][0],
                            dcost[i-1,j-1]+trans_mat[i-1][1],
                            dcost[i-2,j-1]+trans_mat[i-2][2]])
           dcost[i,j] = np.min(costs) +DTW_dist[i-2,j-1]
           tmp_ptr = np.argmin(costs)

           if tmp_ptr == 0:
               DTW_bptr[i,j] = i
           elif tmp_ptr == 1:
               DTW_bptr[i,j] = i-1
           else:
               DTW_bptr[i,j] = i-2
       k=k+2

    np.savetxt('bptr_file',DTW_bptr)

    seg = np.zeros((4,1)) # 4 cuts

    prev=6.0
    current=6.0
    j = n
    btrace = np.zeros((n+1,))
    trans_count = np.zeros((5,5))

    btrace[0] = 2
    btrace[1] = 2
    while j >= 2:
        btrace[j] = prev
        current = DTW_bptr[prev][j]
        j = j - 1
        trans_count[current-2,prev-2] = trans_count[current-2,prev-2] + 1
        prev = current

    btrace = btrace -2
    #print btrace
    #binct = np.bincount(btrace.astype(np.int64,casting='unsafe'))
    binct = np.bincount(btrace.astype(int))
    #print binct
    prev=0
    for j in xrange(4): #Last cut does not matter
            seg[j] = binct[j] + prev
            prev = seg[j]

    tr_count = np.concatenate((np.matrix(trans_count[0,:3]),
                               np.matrix(trans_count[1,1:4]),
                               np.matrix(trans_count[2,2:5]),
                               np.matrix(np.append(trans_count[3,3:],0)),
                               np.matrix(np.append(np.append(trans_count[4,4],0),0))),
                              axis=0)

    tr_count = tr_count + 0.0001 #Avoiding infinity errors
    tr_sum = np.sum(tr_count,axis=1)
    tr_count = tr_count/tr_sum
    #tr_count = tr_count/np.sum(tr_count)
    tr_count = -np.log(tr_count)


    best_cost = dcost[dcost.shape[0]-1,
                      dcost.shape[1]-1]

    return seg, tr_count, best_cost

def train_hmm_isol(digit, data):

    data0 = data
    #print data0.shape
    #segs = np.ones((1,4)) * np.array([0.2,0.4,0.6,0.8])
    #segs = np.array([[data0.shape[0]]]) * segs


    #print segs
    # HMM: Our HMM will be a numpy matrix 10 x 39
    # because 5 states and one row for mean and one row for variance

    # lOAD THIS FROM FILE
    HMM = np.loadtxt(DIRNAME + digit+'.hmm')
    trans_mat = np.loadtxt(DIRNAME+digit+'.trans')

    for i in xrange(30):

        # Do DTW between HMM and data sequence
        new_segs0, new_tr0, best_cost0 = do_DTW(HMM,trans_mat,data0)

        #print new_segs0
        avg_best_cost = best_cost0

        #print 'Iteration',i,' cost: ',avg_best_cost
        # Update rules

        segs = np.concatenate(new_segs0.transpose(),axis=0)
        #print segs
        trans_mat[2:,:] = new_tr0

        # Extract appropriate sections for each state
        state1 = data0[:segs[0]+1,:]
        if state1.shape[0]==1:
            state1 = data0[:segs[0]+2,:]
        #print state1.shape
        HMM[0][:] = np.mean(state1,axis=0)
        HMM[1][:] = np.diag(np.corrcoef(state1, rowvar=0))


        state2 = data0[segs[0]-1:segs[1]+1,:]
        #print state2.shape
        if state2.shape[0]==1:
            state2=data0[segs[0]-2:segs[1]+1,:]
        HMM[2][:] = np.mean(state2,axis=0)
        HMM[3][:] = np.diag(np.corrcoef(state2, rowvar=0))

        state3 = data0[segs[1]-1:segs[2]+1,:]
        if state3.shape[0]==1:
            state3=data0[segs[1]-2:segs[2]+1,:]
        #print state3.shape
        HMM[4][:] = np.mean(state3,axis=0)
        HMM[5][:] = np.diag(np.corrcoef(state3, rowvar=0))


        state4 = data0[segs[2]-1:segs[3]+1,:]
        if state4.shape[0]==1:
            state4=data0[segs[2]-2:segs[3]+1,:]
        #print state4.shape
        HMM[6][:] = np.mean(state4,axis=0)
        HMM[7][:] = np.diag(np.corrcoef(state4, rowvar=0))

        
        state5 = data0[segs[3]-1:,:]
        if state5.shape[0]==1:
            state5=data0[segs[3]-2:,:]
        #print state5.shape
        #print '5th state'
        HMM[8][:] = np.mean(state5,axis=0)
        HMM[9][:] = np.diag(np.corrcoef(state5, rowvar=0))


    filnm = DIRNAME+digit+'.hmm'
    np.savetxt(filnm,HMM)
    filnm = DIRNAME+digit+'.trans'
    np.savetxt(filnm,trans_mat)

    pass



def train_hmm(mapped_symbs, filenm):

    global t_graph
    t_graph = graph()
    hmms=[]

    # Create HMM 
    # for i in xrange(len(mapped_symbs)):
    #     hmms.append(hmm(symbol=mapped_symbs[i]))


    # Read in filenm
    data = np.loadtxt(filenm)
    NO_OF_TIME_SEQ = np.shape(data)[0]

    for i in range(NO_OF_TIME_SEQ):
        for j in range(len(mapped_symbs)):
            t_graph.add_node(template_node(time=i, HMM_no=-1,level_no=j, state_no=0, non_emitting=True, seq=data))
            for k in range(NO_OF_STATES):
                t_graph.add_node(template_node(time=i, HMM_no=int(mapped_symbs[j]),level_no=j, state_no=k, seq=data))
        t_graph.add_node(template_node(time=i, HMM_no=-1, level_no=len(mapped_symbs),state_no=0, non_emitting=True, seq=data))

    result = np.array([])
    timein = np.array([])
    last_frame = t_graph.template_nodes[NO_OF_TIME_SEQ - 1]#[-1]
    curr_nod = last_frame[-1]
    print curr_nod.parents
    while curr_nod.identifier[3] != 0:
        best_par = curr_nod.best_parent
        print '{0}--->{1}---->{2}--->{3}--->{4}'.format( curr_nod.identifier[3],curr_nod.identifier,best_par.identifier,curr_nod.P,curr_nod.C)
        result = np.append(result, curr_nod.identifier[1])
        timein = np.append(timein, curr_nod.identifier[3])
        curr_nod = curr_nod.best_parent

    #print '{0}--->{1}---->{2}---->{3}'.format(curr_nod.identifier[3],curr_nod.identifier,curr_nod.P,curr_nod.C)
    result = np.append(result, curr_nod.identifier[1])
    timein = np.append(timein, curr_nod.identifier[3])
    result = result[::-1]
    timein = timein[::-1]
    boundary = np.where(result == -1)[0]
    timein = timein[boundary]
    timein = np.insert(timein,0,-1)
    #print timein
    segment_list = []
    for i in range(len(mapped_symbs)):
        segment = data[timein[i]+1:timein[i+1]+1, :]
        print '{0}-----{1}'.format(segment.shape,mapped_symbs[i])
        segment_list.append(segment)
    for i in range(len(mapped_symbs)):
        print 'xxxxxxxxxxxxxxxxxxxxxxxx'
        print mapped_symbs[i]
        train_hmm_isol(mapped_symbs[i], segment_list[i])
    

def train_cont_hmms(transcrps, dirname):
    
    #with open(transcrps) as f:
        #trans = f.readlines()
        
    # Chomp trailing newline characters
    #for i in xrange(len(trans)):
        #trans[i] = trans[i].rstrip()
        
        #trans[i] = trans[i].split()
        
        
        #trans[i][-1] = trans[i][-1].replace('(','').replace(')','')
        
    # Read each line of the transcript, compose model and train
    
    #for i in xrange(len(trans)):
    
        # Clean up names
        #mapped_symbs,filenm = cleanup_names(trans[i])
        #print mapped_symbs
        #fullname = dirname + '/' + filenm + '.mfcc'
        
        # Put models together and train
        #train_hmm(mapped_symbs, fullname)
    trans=transcrps.split()
    mapped_symbs,filenam = cleanup_names(trans)
    print mapped_symbs
    print filenam
    fullname = dirname + '/' + filenam+'.mfcc'
    train_hmm(mapped_symbs,fullname)

if __name__ == '__main__':
    # train_cont_hmms(sys.argv[1],sys.argv[2])
    train_cont_hmms(sys.argv[1],sys.argv[2])
