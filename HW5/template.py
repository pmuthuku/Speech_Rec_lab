import copy



__author__ = 'anoop'

NO_OF_TIME_SEQ = 3
NO_OF_LEVELS = 2
NO_OF_HMM = 2
NO_OF_STATES = 3


class graph:
    template_nodes = []
    def add_node(self, template_node):
        self.template_nodes.append(copy.deepcopy(template_node))

    def find_node_by(self, level_no, hmm_no, state_no, time_seq = -1):
        this_level = list(filter(lambda _:_.identifier[0] in level_no, self.template_nodes))
        this_hmm = list(filter(lambda _:_.identifier[1] in hmm_no, this_level))
        this_state = list(filter(lambda _:_.identifier[2] in state_no, this_hmm))
        if(time_seq != -1):
            this_state = list(filter(lambda _:_.identifier[3] == time_seq, this_state))
        return this_state

t_graph = graph()


class template_node:
    def __init__(self, level_no, HMM_no, state_no , time=-1  , non_emitting = False):
        if(time == 0):
            self.parents = []
        elif(level_no == 0 and non_emitting == True):
            self.parents  = []
        elif (non_emitting == True):
            self.parents = t_graph.find_node_by([level_no -1], range(NO_OF_HMM), [NO_OF_STATES - 1], time_seq=time)
        elif(state_no == 0):
            self.parents = t_graph.find_node_by([level_no], [-1, HMM_no], [0], time_seq=time-1)
        elif(state_no == 1):
            self.parents = t_graph.find_node_by([level_no], [HMM_no], range(state_no-1, state_no+1), time_seq=time-1)
        elif(state_no > 1):
            self.parents = t_graph.find_node_by([level_no], [HMM_no], range(state_no-2, state_no+1), time_seq=time-1)
        self.mu = []
        self.sigma = []
        self.non_emitting = non_emitting
        self.identifier = [level_no, HMM_no, state_no, time]
def main():

    for t in range(0, NO_OF_TIME_SEQ):
        for i in range(0, NO_OF_LEVELS):
            t_graph.add_node(template_node(i, -1, 0, time=t, non_emitting=True))
            for j in range(0, NO_OF_HMM):
                for k in range(0, NO_OF_STATES):
                    t_graph.add_node(template_node(i, j, k, time=t))

        t_graph.add_node(template_node(2, -1, 0, time=t, non_emitting=True))
    pass
    t_graph.find_node_by([0],[1],[0])

if __name__=="__main__":
    main()