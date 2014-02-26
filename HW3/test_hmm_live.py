import numpy as np
import scipy.spatial.distance

def do_DTW(HMM, trans_mat, data):
    means = HMM[::2,:]
    vars = HMM[1::2,:]

    #vars = vars+ 0.001

    DTW_dist = np.zeros((5,data.shape[0]))
    
    for i in xrange(5):
        inv_cov = np.linalg.inv(np.diagflat(vars[i][:]))
        tmp_dist = scipy.spatial.distance.cdist(np.matrix(means[i][:]),data,
                                      #          'euclidean')
                                                'mahalanobis',VI=inv_cov)
        DTW_dist[i][:] = 0.5*tmp_dist + 0.5*np.log(np.prod(vars[i][:])) #+ 19.5*np.log(2*np.pi)

    np.savetxt('dist_file',DTW_dist)
                                              

    # Do actual DTW: Anurag's code
    m,n = np.shape(DTW_dist)
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

    binct = np.bincount(btrace.astype(np.int64,casting='unsafe'))


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
    tr_count = tr_count/np.sum(tr_count)
    tr_count = np.log(tr_count)


    best_cost = dcost[dcost.shape[0]-1, 
                      dcost.shape[1]-1]

    return seg, tr_count, best_cost

#tlist=[0,1,2,3,4]
tlist=[5,6,7,8,9]
trlist=range(5,5+1)
        #trlist=range(0,0+int(argv[2]))

        #trl=range(5,10)
        #random.shuffle(trl)
        #trlist=trl[0:int(argv[2])]
ac=0.0


inpdat=np.loadtxt('mel_cep.data')
          
cost=[]
for j in range(0,10):
    hmm=np.loadtxt('all_recs/hmm/'+str(j)+'.hmm')
    tram=np.loadtxt('all_recs/trans/'+str(j)+'.trans')
    r,t,score=do_DTW(hmm,tram,inpdat)
    cost[len(cost):]=[score]
          
          #print cost
          #print '{0}_{1} recognised as {2}'.format(i,l,cost.index(min(cost)))
print cost.index(min(cost))
