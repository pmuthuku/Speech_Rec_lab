import sys
import time
import copy
import numpy as np
import scipy.spatial.distance

np.set_printoptions(threshold='nan', precision=3)

MODEL_DIR_NAME = 'models_a'

#MODEL_DIR_NAME = 'model_euclidean_withconst_cov'
#MODEL_DIR_NAME = 'model_mahalanobis_withconst_cov'
#MODEL_DIR_NAME = 'model_euclidean_withoutconst_cov'
#MODEL_DIR_NAME = 'model_mahalanobis_withoutconst_cov'
#MODEL_DIR_NAME = 'model_euclidean_withconst_corrcoef'
#MODEL_DIR_NAME = 'model_mahalanobis_withconst_corrcoef'
#MODEL_DIR_NAME = 'model_euclidean_withoutconst_corrcoef'
#MODEL_DIR_NAME = 'model_mahalanobis_withoutconst_ccorrcoef'

RUSULTS_FILE_NAME = 'RESULTS/' + MODEL_DIR_NAME + '.result'

audio_file_mfcc_list = [
        'cont_recordings/0_1.mfcc',
        'cont_recordings/0_2.mfcc',
        'cont_recordings/0_3.mfcc',
        'cont_recordings/0_4.mfcc',
        'cont_recordings/0_5.mfcc',#5
        'cont_recordings/1_1.mfcc',
        'cont_recordings/1_2.mfcc',
        'cont_recordings/1_3.mfcc',
        'cont_recordings/1_4.mfcc',
        'cont_recordings/1_5.mfcc',#10
        'cont_recordings/2_1.mfcc',
        'cont_recordings/2_2.mfcc',
        'cont_recordings/2_3.mfcc',
        'cont_recordings/2_4.mfcc',
        'cont_recordings/2_5.mfcc',#15
        'cont_recordings/3_1.mfcc',
        'cont_recordings/3_3.mfcc',
        'cont_recordings/3_4.mfcc',
        'cont_recordings/3_5.mfcc',#19
        'cont_recordings/4_1.mfcc',#20
        'cont_recordings/4_3.mfcc',
        'cont_recordings/4_4.mfcc',#22
        'cont_recordings/4_5.mfcc',#23
        'cont_recordings/5_1.mfcc',#24
        'cont_recordings/5_2.mfcc',#25
        'cont_recordings/5_3.mfcc',#26
        'cont_recordings/5_4.mfcc',#27
        'cont_recordings/5_5.mfcc'#28
]

correct_pronounce = [
    '5 3 0 4 8 4 6 4 0 2',
    '4 4 6 3 7 1 7 4 0 0',
    '8 3 0 1 2 6 9 4 5 6',
    '7 4 8 6 3 8 3 3 9 0',
    '4 4 9 9 2 6 6 3 2 5',#5
    '8 4 6 1 1 9 5 2 5 8',
    '3 2 6 8 0 6 0 0 7 4',
    '9 7 8 6 0 1 0 2 6 1',
    '7 2 2 1 0 4 6 0 0 3',
    '2 7 9 2 4 8 2 1 7 8',#10
    '8 8 9 2 6 3 7 7 3 8',
    '5 2 4 5 1 9 5 4 1 1',
    '1 5 3 2 1 6 1 3 1 5',
    '3 7 6 0 8 0 2 6 4 3',
    '6 7 2 8 9 8 8 3 5 2',#15
    '8 2 8 2 4 2 0 0 1 2',
    '1 0 8 8 3 6 5 1 5 8',
    '4 2 3 6 9 6 2 7 2 3',
    '5 6 1 1 5 0 3 9 5 9',#19
    '6 5 0 1 7 2 4 1 2 3',#20
    '5 9 5 5 4 7 3 4 5 4',
    '9 1 6 9 5 6 1 9 6 3',#22
    '3 8 5 5 5 7 6 1 0 9',
    '3 1 7 1 4 3 4 5 1 4',
    '2 3 7 0 9 9 3 0 9 0',
    '2 0 3 1 5 7 2 4 9 8',
    '1 8 1 6 2 1 7 0 6 5',
    '4 9 9 4 1 5 1 9 3 4',
]

# NO_OF_TIME_SEQ = 3
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
               MODEL_DIR_NAME + '/9.trans',
               MODEL_DIR_NAME + '/sil.trans'
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
             MODEL_DIR_NAME + '/9.hmm',
             MODEL_DIR_NAME + '/sil.hmm'
];

mu_list = [np.zeros((NO_OF_STATES, 39)) for x in range(NO_OF_HMM + 1)]
sigma_list = [np.zeros((NO_OF_STATES, 39)) for x in range(NO_OF_HMM + 1)]

trans_list = []

for hmm_idx, hmm_file in enumerate(hmm_names):
    f = np.loadtxt(hmm_file)
    for i in range(NO_OF_STATES):
        mu_list[hmm_idx][i] = f[i * 2]
        sigma_list[hmm_idx][i] = f[i * 2 + 1]

for trns_idx, trans_file in enumerate(trans_names):
    f = np.loadtxt(trans_file)
    trans = np.empty((NO_OF_STATES, NO_OF_STATES))
    #trans.fill(-1 * np.inf)
    trans.fill(np.inf)
    for j in range(NO_OF_STATES):
        if j > 2:
            subst = f[2 + j, 0:5 - j]
        else:
            subst = f[2 + j, :]

        trans[j, j:min(j + 3, NO_OF_STATES)] = subst#np.delete(f[2+j, :], np.where(f[2+j, :] == np.inf), axis=0)
    trans_list[len(trans_list):] = [trans]

input_seq = None


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


t_graph = None


def calculate_C(xn, mu, sigma):
    inv_cov = np.linalg.inv(np.diagflat(sigma))
    tmp_dist = scipy.spatial.distance.cdist(np.matrix(mu), np.matrix(xn),
                                            #          'euclidean')
                                            'mahalanobis', VI=inv_cov)
    C = (0.5 * tmp_dist)+(0.5 * np.log(np.prod(sigma)))+(19.5 * np.log(2 * np.pi))
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
            trans_cost = np.log(0.5)
        elif my_state_no == 0 and my_hmm_no == 10:
            trans_cost = np.log(0.5)
        elif parent_hmm_no == -1:
            trans_cost = np.log(0.1)
        else :
            trans_cost = trans_list[my_hmm_no][parent_state_no, my_state_no]

        tr_cost = parent.P + trans_cost
        if tr_cost < best_P_prev:
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
            self.parents = t_graph.find_node_by([level_no - 1], range(NO_OF_HMM+1), [NO_OF_STATES - 1], time_seq=time)
        elif(state_no == 0 and HMM_no == 10):
            self.parents = t_graph.find_node_by([level_no],range(HMM_no), [NO_OF_STATES-1], time_seq=time-1 )
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
        if (HMM_no != -1):
            if time == 0:
                if (state_no == 0 and (level_no == 0 and HMM_no !=10)):
                    C = calculate_C(input_seq[time, :], self.mu, self.sigma)
                else:
                    C = np.inf
                    #C = -1 * np.inf
            else:
                C = calculate_C(input_seq[time, :], self.mu, self.sigma)

        else:
            if level_no == 0: #for third problem make it level_no ==0 and time==0
                #C = -1 * np.inf
                C = np.inf
            elif time == 0:
                #C = -1 * np.inf
                C = np.inf
            else:
                C = 0
        self.C = C
        (self.P, self.best_parent) = calculate_P(self.parents, self.identifier, self.C)


# def main(file_name='ph_nos/0_1.mfcc', file_number=0):
def main(file_name='cont_recordings/4_3.mfcc', file_number=20):
    global t_graph
    t_graph = graph()
    global input_seq

    fp = open(RUSULTS_FILE_NAME, 'w')

    input_seq = np.loadtxt(file_name)#.transpose()
    NO_OF_TIME_SEQ = input_seq.shape[0]
    # NO_OF_TIME_SEQ = 50

    kk = 0
    result = np.array([])
    for t in range(0, NO_OF_TIME_SEQ):
        #print t
        st = time.time()
        for i in range(0, NO_OF_LEVELS):
            t_graph.add_node(template_node(i, -1, 0, time=t, non_emitting=True))
            for j in range(0, NO_OF_HMM):
                for k in range(0, NO_OF_STATES):
                    t_graph.add_node(template_node(i, j, k, time=t))
                    kk = kk + 1
            for k in range(0, NO_OF_STATES):
                t_graph.add_node(template_node(i, 10, k, time=t))   #HMM no_10 for silence
        t_graph.add_node(template_node(NO_OF_LEVELS, -1, 0, time=t, non_emitting=True))
        et = time.time()

    tt = NO_OF_TIME_SEQ
    last_frame = t_graph.template_nodes[NO_OF_TIME_SEQ - 1]#[-1]
    curr_nod = last_frame[-1]
    #curr_nod_parents=curr_nod.parents
    #print curr_nod.identifier
    #print 'XXXXXXXXXXXXXXXX'
    #for pt in curr_nod_parents:
    #    print pt.identifier
    while curr_nod.identifier[3] != 0:
        best_par=curr_nod.best_parent
        #print '{0}--->{1}---->{2}--->{3}--->{4}'.format( curr_nod.identifier[3],curr_nod.identifier,best_par.identifier,curr_nod.P,curr_nod.C)
        #if (result[-1] != curr_nod.identifier[1]): result = np.append(result, curr_nod.identifier[1])
        result = np.append(result, curr_nod.identifier[1])
        curr_nod=curr_nod.best_parent
        tt=tt-1

    #print '{0}--->{1}---->{2}---->{3}'.format(curr_nod.identifier[3],curr_nod.identifier,curr_nod.P,curr_nod.C)
    #first_frame=t_graph.template_nodes[0]
    #for nod in first_frame:
    #    print '{0}--->{1}---->{2}--->{3}'.format( nod.identifier[3],nod.identifier,nod.P,nod.C)
    
    result = np.append(result, curr_nod.identifier[1])
    #resultid = np.where(result != -1)[0]
    # print np.shape(resultid)
    #resultid = resultid + 1
    #result = result[resultid]
    #indx=np.unique(result,return_index=True)[1]
    #fresult = [result[idx] for idx in sorted(indx)]
    # result = result[::-1]
    # result = map(int, result)
    #result = np.delete(result, np.where(result == -1))[::-1]
    prev=result[0]
    #print prev
    fresult=np.array([prev])
    for rs in result:
        if rs != prev:
            #print rs
            fresult=np.append(fresult,rs)
            prev=rs

    fresultid=np.where(fresult !=-1.0)[0]
    fresult=fresult[fresultid]
    fresultid=np.where(fresult !=10.0)[0]
    fresult=fresult[fresultid]
    fresult=fresult[::-1]
    fresult = map(int,fresult)
    resultf=''
    for x in fresult:
        resultf=resultf+str(x)+' '

    print 'Original -- > {0}'.format(correct_pronounce[file_number])
    print 'Detected -- > {0}'.format(resultf)
   
    fp.write(file_name + '\n     Recognized a   s   -->     ' + str(result) + '\n     Correct Result  -->     ' +
             correct_pronounce[file_number] + '\n')
    fp.close()


if __name__ == "__main__":
    #fp = open(RUSULTS_FILE_NAME, 'w')
    #fp.close()

    #for audio_file_number, audio_file in enumerate(audio_file_mfcc_list):
    #print "Running"
    audio_file_number=int(sys.argv[1])
    audio_file=sys.argv[2]
    # print audio_file_number
    # print audio_file
    main(audio_file,audio_file_number)
