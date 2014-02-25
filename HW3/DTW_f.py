import sys
import numpy as np
import scipy.spatial.distance
import time
if len(sys.argv) <= 2:
    print "Usage:\npython DTW.py input template\n"
    exit(0)


def get_on_nodes(val_nodes):
    
    current_on_nodes = val_nodes.nonzero()
    current_on_nodes = current_on_nodes[0]

    #Initialize active nodes
    active_nodes = np.zeros((val_nodes.shape[0],), dtype=np.int8)

    for k in current_on_nodes[:]:
        active_nodes[k] = 1
        if (k+1 < val_nodes.shape[0]):
            active_nodes[k+1] = 1
        if (k+2 < val_nodes.shape[0]):
            active_nodes[k+2] = 1
	
	active_no=active_nodes.nonzero()
    return active_no[0]
    
st=time.clock()
input_data = np.loadtxt(sys.argv[1])
template_data = np.loadtxt(sys.argv[2])

# Initialize DTW cost matrix
DTW_costs = np.zeros((template_data.shape[0], input_data.shape[0]))
DTW_costs = DTW_costs + np.inf


# Calculate distance matrix
DTW_distance = scipy.spatial.distance.cdist(template_data, input_data,'euclidean')

# Valid set a.k.a unpruned nodes
valid_nodes = np.zeros((template_data.shape[0],), dtype=np.int8)
active_nodes = np.array([0])
valid_nodes[0]=1
for i in xrange(input_data.shape[0]):

    for j in active_nodes[:]:
        
        comp_cost = DTW_distance[j][i]

        # Let's make an array to find the minimum costs
        if (i==0):
            costs = [comp_cost , np.inf, np.inf]
                    # horiz | diag | super-diag

        elif(j==0):
            costs = [DTW_costs[j][i-1] + comp_cost,
                     np.inf, np.inf]

        elif(j==1):
            costs = [DTW_costs[j][i-1] + comp_cost,
                     DTW_costs[j-1][i-1] + comp_cost,
                     np.inf]
        else:
            costs = [DTW_costs[j][i-1] + comp_cost,
                     DTW_costs[j-1][i-1] + comp_cost,
                     DTW_costs[j-2][i-1] + comp_cost]

        DTW_costs[j][i] = np.min(costs)
	valid_nodes[active_nodes]=1	
	active_nodes = get_on_nodes(valid_nodes)

et=time.clock()
print (et-st)
print DTW_costs[DTW_costs.shape[0]-1][DTW_costs.shape[1]-1]
