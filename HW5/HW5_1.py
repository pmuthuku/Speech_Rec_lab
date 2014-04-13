import time
import copy
import numpy as np
np.set_printoptions(threshold='nan', precision=3)

#NO_OF_TIME_SEQ = 3
NO_OF_LEVELS = 10
NO_OF_HMM = 10
NO_OF_STATES = 5

#input_seq = [np.zeros((39,1)) for _ in range(NO_OF_TIME_SEQ)]
input_seq = np.loadtxt('new_recordings/anurag_1.mfcc')#.transpose()
#input_seq = np.asarray(input_seq)
#input_seq.shape
NO_OF_TIME_SEQ = input_seq.shape[0]
#print NO_OF_TIME_SEQ
#pass


trans_names = ['models/0.trans',
               'models/1.trans',
               'models/2.trans',
               'models/3.trans',
               'models/4.trans',
               'models/5.trans',
               'models/6.trans',
               'models/7.trans',
               'models/8.trans',
               'models/9.trans'
];
hmm_names = ['models/0.hmm',
             'models/1.hmm',
             'models/2.hmm',
             'models/3.hmm',
             'models/4.hmm',
             'models/5.hmm',
             'models/6.hmm',
             'models/7.hmm',
             'models/8.hmm',
             'models/9.hmm'
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
        subst=f[2+j,np.where(f[2+j,:]!=np.inf)]
        trans[j, j:min(j+3, NO_OF_STATES)] = subst#np.delete(f[2+j, :], np.where(f[2+j, :] == np.inf), axis=0)
    #print trans
    trans_list[len(trans_list):]=[trans]

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

t_graph = graph()

def calculate_C(xn, mu, sigma):
    # TODO: Assume sigma is not squared while saving
    # sigma_sqr = sigma     --> In case sigma squared was stored instead of sigma


    #if not working try the numpy cdist calculation
    sigma_sqr = sigma # np.square(sigma)
    term1 = np.sum(np.log(sigma_sqr * 2 * np.math.pi)) * (-0.5)
    term2 = np.sum(np.divide(np.square(xn - mu), sigma_sqr)) * (-0.5)
    C = term1 + term2
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
            if level_no == 0:
                C = -1*np.inf
            elif time == 0:
                C = -1*np.inf
            else:
                C = 0
        self.C = C
        (self.P, self.best_parent) = calculate_P(self.parents, self.identifier, self.C)





def main():
    kk=0
    for t in range(0, NO_OF_TIME_SEQ):
        print t
        st=time.time()
        for i in range(0, NO_OF_LEVELS):
            t_graph.add_node(template_node(i, -1, 0, time=t, non_emitting=True))
            for j in range(0, NO_OF_HMM):
                for k in range(0, NO_OF_STATES):
                    t_graph.add_node(template_node(i, j, k, time=t))
                    kk=kk+1
        t_graph.add_node(template_node(NO_OF_LEVELS, -1, 0, time=t, non_emitting=True))
        et=time.time()
        print et-st
    t_graph.find_node_by([0], [1], [0])
    
    print kk
    #print t_graph.template_nodes
    for time_seq,list_at_time_seq in enumerate(t_graph.template_nodes):
        for node in list_at_time_seq:
            if len(node.parents)==0:
                print '{0}---{1}--empty---{2}--{3}'.format(time_seq,node.identifier, node.C, node.P)
            else:
                for pr in node.parents:
                    print '{0}---{1}--{2}---{3}---{4}'.format(time_seq,node.identifier,pr.identifier,node.C,node.P)

    tt=NO_OF_TIME_SEQ
    curr_nod=t_graph.template_nodes[NO_OF_TIME_SEQ - 1][-1]
    while tt-1 >= 0:
        best_par=curr_nod.best_parent
        print '{0}----{1}'.format(curr_nod.identifier,best_par.identifier)
        curr_nod=curr_nod.best_parent
        tt=tt-1


if __name__ == "__main__":
    main()
