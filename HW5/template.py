import copy
import numpy as np
np.set_printoptions(threshold='nan', precision=3)

NO_OF_TIME_SEQ = 2
NO_OF_LEVELS = 2
NO_OF_HMM = 10
NO_OF_STATES = 5

input_seq = np.zeros((39,NO_OF_TIME_SEQ))
# input_seq = np.loadtxt('new_recordings/anoop_1.mfcc').transpose()
# input_seq = np.asarray(input_seq)
NO_OF_TIME_SEQ = input_seq.shape[1]
pass


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

trans = np.empty((NO_OF_STATES, NO_OF_STATES))
trans.fill(np.inf)
trans_list = [copy.deepcopy(trans) for x in range(NO_OF_HMM)]

for hmm_idx, hmm_file in enumerate(hmm_names):
    f = np.loadtxt(hmm_file)
    for i in range(NO_OF_STATES):
        mu_list[hmm_idx][i] = f[i*2]
        sigma_list[hmm_idx][i] = f[i*2 +1]


for trns_idx, trans_file in enumerate(trans_names):
    f = np.loadtxt(trans_file)
    
    # truncated_trans = f[2:, :]
    for j in range(NO_OF_STATES):
        x=np.delete(f[2+j, :], np.where(f[2+j, :] == np.inf), axis=0)
        print x
        trans_list[trns_idx][j, j:min(j+3, NO_OF_STATES)] = np.delete(f[2+j, :], np.where(f[2+j, :] == np.inf), axis=0)

class graph:
    template_nodes = []

    def add_node(self, template_node):
        self.template_nodes.append(copy.deepcopy(template_node))

    def find_node_by(self, level_no, hmm_no, state_no, time_seq=-1):
        this_level = list(filter(lambda _: _.identifier[0] in level_no, self.template_nodes))
        this_hmm = list(filter(lambda _: _.identifier[1] in hmm_no, this_level))
        this_state = list(filter(lambda _: _.identifier[2] in state_no, this_hmm))
        if (time_seq != -1):
            this_state = list(filter(lambda _: _.identifier[3] == time_seq, this_state))
        return this_state


t_graph = graph()

def calculate_C(xn, mu, sigma):
    # TODO: Assume sigma is not squared while saving
    # sigma_sqr = sigma     --> In case sigma squared was stored instead of sigma
    sigma_sqr = np.square(sigma)
    term1 = np.sum(np.log(sigma_sqr * 2 * np.math.pi)) * (-0.5)
    term2 = np.sum(np.divide(np.square(xn - mu), sigma_sqr)) * (-0.5)
    C = term1 + term2
    return  C

def calculate_P(parents, identifier, C):
    if len(parents) == 0:
        return (C, None)

    hmm_no = identifier[1]
    my_state_no = identifier[2]

    best_P_prev = -np.inf
    path_idx = -1
    for par_idx, parent in enumerate(parents):
        parent_state_no = parent.identifier[2]
        tr_cost = parent.P +  trans_list[hmm_no][parent_state_no, my_state_no]
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
        #print len(self.parents)
        # Compute C
        if(HMM_no != -1):
            C = calculate_C(input_seq[:, time], self.mu, self.sigma)
        else: C = 0
        self.C = C

        (self.P, self.best_parent) = calculate_P(self.parents, self.identifier, self.C)





def main():
    for t in range(0, NO_OF_TIME_SEQ):
        print t
        for i in range(0, NO_OF_LEVELS):
            t_graph.add_node(template_node(i, -1, 0, time=t, non_emitting=True))
            for j in range(0, NO_OF_HMM):
                for k in range(0, NO_OF_STATES):
                    t_graph.add_node(template_node(i, j, k, time=t))

        t_graph.add_node(template_node(2, -1, 0, time=t, non_emitting=True))
    t_graph.find_node_by([0], [1], [0])


if __name__ == "__main__":
    main()
