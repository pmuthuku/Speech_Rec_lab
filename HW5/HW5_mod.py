import sys
import time
import copy
import numpy as np
import scipy.spatial.distance
np.set_printoptions(threshold='nan', precision=3)



MODEL_DIR_NAME = 'models_new_2'
#MODEL_DIR_NAME = 'model_euclidean_withconst_cov'
#MODEL_DIR_NAME = 'model_mahalanobis_withconst_cov'
#MODEL_DIR_NAME = 'model_euclidean_withoutconst_cov'
# MODEL_DIR_NAME = 'model_mahalanobis_withoutconst_cov'
#MODEL_DIR_NAME = 'model_euclidean_withconst_corrcoef'
# MODEL_DIR_NAME = 'model_mahalanobis_withconst_corrcoef'
#MODEL_DIR_NAME = 'model_euclidean_withoutconst_corrcoef'
#MODEL_DIR_NAME = 'model_mahalanobis_withoutconst_ccorrcoef'

RUSULTS_FILE_NAME =  'RESULTS/' + MODEL_DIR_NAME + '.result'

audio_file_mfcc_list = [
        'ph_nos/0_1.mfcc',
        'ph_nos/0_2.mfcc',
        'ph_nos/0_3.mfcc',
        'ph_nos/0_4.mfcc',
        'ph_nos/0_5.mfcc',
        'ph_nos/1_1.mfcc',
        'ph_nos/1_2.mfcc',
        'ph_nos/1_3.mfcc',
        'ph_nos/1_4.mfcc',
        'ph_nos/1_5.mfcc',
        'ph_nos/2_1.mfcc',
        'ph_nos/2_2.mfcc',
        'ph_nos/2_3.mfcc',
        'ph_nos/2_4.mfcc',
        'ph_nos/2_5.mfcc',
        'ph_nos/3_1.mfcc',
        'ph_nos/3_2.mfcc',
        'ph_nos/3_3.mfcc',
        'ph_nos/3_4.mfcc',
        'ph_nos/3_5.mfcc',
        'ph_nos/4_1.mfcc',
        'ph_nos/4_2.mfcc',
        'ph_nos/4_3.mfcc',
        'ph_nos/4_4.mfcc',#24
        'ph_nos/4_5.mfcc',#
        'ph_nos/5_1.mfcc',#26
        'ph_nos/5_2.mfcc',#27
        'ph_nos/5_3.mfcc',
        'ph_nos/5_4.mfcc',
        'ph_nos/5_5.mfcc'
]

correct_pronounce = [
    '[5 3 0 4 8 4 6 4 0 2]',
    '[4 4 6 3 7 1 7 4 0 0]',
    '[8 3 0 1 2 6 9 4 5 6]',
    '[7 4 8 6 3 8 3 3 9 0]',
    '[4 4 9 9 2 6 6 3 2 5]',
    '[8 4 6 1 1 9 5 2 5 8]',
    '[3 2 6 8 0 6 0 0 7 4]',
    '[9 7 8 6 0 1 0 2 6 1]',
    '[7 2 2 1 0 4 6 0 0 3]',
    '[2 7 9 2 4 8 2 1 7 8]',
    '[8 8 9 2 6 3 7 7 3 8]',
    '[5 2 4 5 1 9 5 4 1 1]',
    '[1 5 3 2 1 6 1 3 1 5]',
    '[3 7 6 0 8 0 2 6 4 3]',
    '[6 7 2 8 9 8 8 3 5 2]',
    '[8 2 8 2 4 2 0 0 1 2]',
    '[3 5 7 3 0 2 6 5 5 3]',
    '[1 0 8 8 3 6 5 1 5 8]',
    '[4 2 3 6 9 6 2 7 2 3]',
    '[5 6 1 1 5 0 3 9 5 9]',
    '[6 5 0 1 7 2 4 1 2 3]',
    '[2 5 5 1 1 6 5 8 8 4]',
    '[5 9 5 5 4 7 3 4 5 4]',
    '[9 1 6 9 5 6 1 9 6 3]',
    '[3 8 5 5 5 7 6 1 0 9]',
    '[3 1 7 1 4 3 4 5 1 4]',
    '[2 3 7 0 9 9 3 0 9 0]',
    '[2 0 3 1 5 7 2 4 9 8]',
    '[1 8 1 6 2 1 7 0 6 5]',
    '[4 9 9 4 1 5 1 9 3 4]',
]


#NO_OF_TIME_SEQ = 3
NO_OF_LEVELS = 10
NO_OF_HMM = 10
NO_OF_STATES = 5



trans_names = [MODEL_DIR_NAME + '/0.trans',
               MODEL_DIR_NAME + '/1.trans',
               MODEL_DIR_NAME + '/2.trans',
               MODEL_DIR_NAME + '/3.trans',
               MODEL_DIR_NAME + '/4.trans',
               MODEL_DIR_NAME + '/5.trans',
               MODEL_DIR_NAME + '/6.trans',
               MODEL_DIR_NAME + '/7.trans',
               MODEL_DIR_NAME + '/8.trans',
               MODEL_DIR_NAME + '/9.trans'
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
             MODEL_DIR_NAME + '/9.hmm'
];

mu_list = [np.zeros((NO_OF_STATES,39)) for x in range(NO_OF_HMM)]
sigma_list = [np.zeros((NO_OF_STATES, 39)) for x in range(NO_OF_HMM)]

#trans = np.empty((NO_OF_STATES, NO_OF_STATES))
#trans.fill(np.inf)
#trans_list = [copy.deepcopy(trans) for x in range(NO_OF_HMM)]
trans_list=[]
for hmm_idx, hmm_file in enumerate(hmm_names):
    f = np.loadtxt(hmm_file)
    for i in range(NO_OF_STATES):
        mu_list[hmm_idx][i] = f[i*2]
        sigma_list[hmm_idx][i] = f[i*2 +1]


for trns_idx, trans_file in enumerate(trans_names):
    f = np.loadtxt(trans_file)
    trans = np.empty((NO_OF_STATES, NO_OF_STATES))
    trans.fill(-1*np.inf)
    # truncated_trans = f[2:, :]
    for j in range(NO_OF_STATES):
        #x=np.delete(f[2+j, :], np.where(f[2+j, :] == np.inf), axis=0)
        #x.shape
        #y=trans_list[trns_idx][j, j:min(j+3, NO_OF_STATES)]
        #y.shape
        #trans_list[trns_idx][j, j:min(j+3, NO_OF_STATES)] = np.delete(f[2+j, :], np.where(f[2+j, :] == np.inf), axis=0)
        if j > 2:
            subst=f[2+j,0:5-j]
        else:
            subst=f[2+j,:]

        trans[j, j:min(j+3, NO_OF_STATES)] = subst#np.delete(f[2+j, :], np.where(f[2+j, :] == np.inf), axis=0)
    #print trans
    trans_list[len(trans_list):]=[trans]

input_seq = None

#for tr in trans_list:
#    print tr
#    print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
#print trans_list
class graph:
    template_nodes = []

    def add_node(self, template_node):
        #self.template_nodes.append(copy.deepcopy(template_node))
        #self.template_nodes.append(template_node)
        # self.template_nodes[len(self.template_nodes):]=[template_node]
        if(len(self.template_nodes) <= template_node.identifier[3]):
            self.template_nodes.append([])
        self.template_nodes[template_node.identifier[3]].append(template_node)
    def find_node_by(self, level_no, hmm_no, state_no, time_seq=-1):
        this_time = self.template_nodes[time_seq]

        this_level = list(filter(lambda _: _.identifier[0] in level_no, this_time))
        this_hmm = list(filter(lambda _: _.identifier[1] in hmm_no, this_level))
        this_state = list(filter(lambda _: _.identifier[2] in state_no, this_hmm))
        return this_state

t_graph = None

def calculate_C(xn, mu, sigma):
    #sigma_sqr = sigma # np.square(sigma)
    #term1 = np.sum(np.log(sigma_sqr * 2 * np.math.pi)) * (-0.5)
    #term2 = np.sum(np.divide(np.square(xn - mu), sigma_sqr)) * (-0.5)
    #C = term1 + term2
    inv_cov = np.linalg.inv(np.diagflat(sigma))
    tmp_dist = scipy.spatial.distance.cdist(np.matrix(mu),np.matrix(xn),
                                      #          'euclidean')
                                                'mahalanobis',VI=inv_cov)
    C = -(0.5*tmp_dist)-(0.5*np.log(np.prod(sigma)))-(19.5*np.log(2*np.pi))
    return  C

def calculate_P(parents, identifier, C):
    if len(parents) == 0:
        return (C, None)

    my_hmm_no = identifier[1]
    my_state_no = identifier[2]

    best_P_prev = -np.inf
    path_idx = -1
    for par_idx, parent in enumerate(parents):
        parent_state_no = parent.identifier[2]
        parent_hmm_no = parent.identifier[1]
        if parent_hmm_no == -1 or my_hmm_no == -1: # if parent is non-emitting then transition probability log is 0 but check insta. trans is from emitt to non-emmiting. Non-emiitting to emmiting is computed on next frame
            trans_cost= 0
        else:
            trans_cost= trans_list[my_hmm_no][parent_state_no, my_state_no]

        tr_cost = parent.P + trans_cost
        if tr_cost > best_P_prev:
            best_P_prev = tr_cost
            path_idx = par_idx

    return (best_P_prev + C, parents[path_idx])


class template_node:

    def __init__(self, level_no, HMM_no, state_no, time=-1, non_emitting=False):
        if (time == 0):
            self.parents = []
        elif (level_no == 0 and non_emitting == True):
            self.parents = []
        elif (non_emitting == True):
            self.parents = t_graph.find_node_by([level_no - 1], range(NO_OF_HMM), [NO_OF_STATES - 1], time_seq=time)
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
                    C = calculate_C(input_seq[time,:], self.mu, self.sigma)
                else:
                    C=-1*np.inf
            else:
                C = calculate_C(input_seq[time,:], self.mu, self.sigma)
        else:
            if level_no == 0: #for third problem make it level_no ==0 and time==0
                C = -1*np.inf
            elif time == 0:
                C = -1*np.inf
            else:
                C = 0
        self.C = C
        (self.P, self.best_parent) = calculate_P(self.parents, self.identifier, self.C)





def main(file_name='new_recordings/anurag_2.mfcc', file_number = 0):
    global t_graph
    t_graph = graph()
    global input_seq

    fp = open(RUSULTS_FILE_NAME, 'a')

    #input_seq = [np.zeros((39,1)) for _ in range(NO_OF_TIME_SEQ)]
    input_seq = np.loadtxt(file_name)#.transpose()
    #input_seq = np.asarray(input_seq)
    #input_seq.shape
    NO_OF_TIME_SEQ = input_seq.shape[0]
    #print NO_OF_TIME_SEQ
    #pass


    kk=0
    result = np.array([])
    for t in range(0, NO_OF_TIME_SEQ):
        #print t
        st=time.time()
        for i in range(0, NO_OF_LEVELS):
            t_graph.add_node(template_node(i, -1, 0, time=t, non_emitting=True))
            for j in range(0, NO_OF_HMM):
                for k in range(0, NO_OF_STATES):
                    t_graph.add_node(template_node(i, j, k, time=t))
                    kk=kk+1
        t_graph.add_node(template_node(NO_OF_LEVELS, -1, 0, time=t, non_emitting=True))
        et=time.time()
        #print et-st
    # t_graph.find_node_by([0], [1], [0])
    #
    # print kk
    #print t_graph.template_nodes
    #for time_seq,list_at_time_seq in enumerate(t_graph.template_nodes):
    #    for node in list_at_time_seq:
    #        if len(node.parents)==0:
    #            print '{0}---{1}--empty---{2}--{3}'.format(time_seq,node.identifier, node.C, node.P)
    #        else:
    #            for pr in node.parents:
    #                print '{0}---{1}--{2}---{3}---{4}'.format(time_seq,node.identifier,pr.identifier,node.C,node.P)

    tt=NO_OF_TIME_SEQ
    last_frame=t_graph.template_nodes[NO_OF_TIME_SEQ - 1]#[-1]
    curr_nod = last_frame[-1]
    while curr_nod.identifier[3] != 0:
        best_par=curr_nod.best_parent
        #print '{0}--->{1}---->{2}--->{3}--->{4}'.format( curr_nod.identifier[3],curr_nod.identifier,best_par.identifier,curr_nod.P,curr_nod.C)
        #if (result[-1] != curr_nod.identifier[1]): result = np.append(result, curr_nod.identifier[1])
        result = np.append(result, curr_nod.identifier[1])
        curr_nod=curr_nod.best_parent
        tt=tt-1

    #print '{0}--->{1}---->{2}---->{3}'.format(curr_nod.identifier[3],curr_nod.identifier,curr_nod.P,curr_nod.C)
    result = np.append(result, curr_nod.identifier[1])
    resultid = np.where(result==-1)[0]
    print np.shape(resultid)
    resultid=resultid+1
    result=result[resultid]
    result=result[::-1]
    result=map(int,result)
    #result = np.delete(result, np.where(result == -1))[::-1]
    print '\n\nYou Said -- > '
    print result
    print '\n Original - {0}'.format(correct_pronounce[file_number])
    fp.write(file_name + '\n     Recognized as   -->     ' + str(result) + '\n     Correct Result  -->     '+correct_pronounce[file_number] + '\n')
    fp.close()

if __name__ == "__main__":
    #fp = open(RUSULTS_FILE_NAME, 'w')
    #fp.close()

    #for audio_file_number, audio_file in enumerate(audio_file_mfcc_list):
    print "Running"
    audio_file_number=int(sys.argv[1])
    audio_file=sys.argv[2]
    print audio_file_number
    print audio_file
    main(audio_file, audio_file_number)
