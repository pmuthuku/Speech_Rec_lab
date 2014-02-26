import sys
import numpy as np
import scipy.spatial.distance

def do_DTW(HMM, data):
    means = HMM[::2,:]
    vars = HMM[1::2,:]

    DTW_dist = np.zeros((5,data.shape[0]))
    
    for i in xrange(5):
        inv_cov = np.linalg.inv(np.diagflat(vars[i][:]))
        DTW_dist[i][:] = scipy.spatial.distance.cdist(np.matrix(means[i][:]),data,
                                                      'mahalanobis',VI=inv_cov)

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
           dcost[i,j] = min(dcost[i,j-1],dcost[i-1,j-1],dcost[i-2,j-1])+DTW_dist[i-2,j-1]
           tmp_ptr = np.argmin([dcost[i,j-1],dcost[i-1,j-1],dcost[i-2,j-1]])

           if tmp_ptr == 0:
               DTW_bptr[i,j] = i
           elif tmp_ptr == 1:
               DTW_bptr[i,j] = i-1
           else:
               DTW_bptr[i,j] = i-2
       k=k+2

    #np.savetxt('bptr_file',DTW_bptr)

    seg = np.zeros((4,1)) # 4 cuts

    prev=6.0
    current=6.0
    j = n
    k = 3
    while current > 2.0:
        current = DTW_bptr[prev][j]
        j = j - 1
        
        if current != prev: #State has changed
            seg[k] = j
            k = k-1
             
        prev = current
    
    pass



if len(sys.argv) <= 1:
    print "Usage:\npython train_hmm.py digit"
    print "Run in directory with MFCCs\n"
    exit(0)

def train_hmm(digit):

    data0 = np.loadtxt(digit+'_0.mfcc')
    data1 = np.loadtxt(digit+'_1.mfcc')
    data2 = np.loadtxt(digit+'_2.mfcc')
    data3 = np.loadtxt(digit+'_3.mfcc')
    data4 = np.loadtxt(digit+'_4.mfcc')

    segs = np.ones((5,4)) * np.array([0.2,0.4,0.6,0.8])
    segs = np.array([[data0.shape[0]],
                     [data1.shape[0]],
                     [data2.shape[0]],
                     [data3.shape[0]],
                     [data4.shape[0]]]) * segs

    # HMM: Our HMM will be a numpy matrix 10 x 39
    # because 5 states and one row for mean and one row for variance
    HMM = np.zeros((10,data0.shape[1]))

    
         
    # Extract appropriate sections for each state
    state1 = np.concatenate((data0[:segs[0][0],:],
                             data1[:segs[1][0],:],
                             data2[:segs[2][0],:],
                             data3[:segs[3][0],:],
                             data4[:segs[4][0],:]),axis=0)

    HMM[0][:] = np.mean(state1,axis=0)
    HMM[1][:] = np.diag(np.cov(state1, rowvar=0))



    state1 = np.concatenate((data0[segs[0][0]:segs[0][1],:],
                             data1[segs[1][0]:segs[1][1],:],
                             data2[segs[2][0]:segs[2][1],:],
                             data3[segs[3][0]:segs[3][1],:],
                             data4[segs[4][0]:segs[4][1],:]),axis=0)

    HMM[2][:] = np.mean(state1,axis=0)
    HMM[3][:] = np.diag(np.cov(state1, rowvar=0))
    


    state1 = np.concatenate((data0[segs[0][1]:segs[0][2],:],
                             data1[segs[1][1]:segs[1][2],:],
                             data2[segs[2][1]:segs[2][2],:],
                             data3[segs[3][1]:segs[3][2],:],
                             data4[segs[4][1]:segs[4][2],:]),axis=0)

    HMM[4][:] = np.mean(state1,axis=0)
    HMM[5][:] = np.diag(np.cov(state1, rowvar=0))



    state1 = np.concatenate((data0[segs[0][2]:segs[0][3],:],
                             data1[segs[1][2]:segs[1][3],:],
                             data2[segs[2][2]:segs[2][3],:],
                             data3[segs[3][2]:segs[3][3],:],
                             data4[segs[4][2]:segs[4][3],:]),axis=0)

    HMM[6][:] = np.mean(state1,axis=0)
    HMM[7][:] = np.diag(np.cov(state1, rowvar=0))
    


    state1 = np.concatenate((data0[segs[0][3]:,:],
                             data1[segs[1][3]:,:],
                             data2[segs[2][3]:,:],
                             data3[segs[3][3]:,:],
                             data4[segs[4][3]:,:]),axis=0)

    HMM[8][:] = np.mean(state1,axis=0)
    HMM[9][:] = np.diag(np.cov(state1, rowvar=0))

    # Do DTW between HMM and data sequence
    do_DTW(HMM,data0)
    do_DTW(HMM,data1)
    do_DTW(HMM,data2)
    do_DTW(HMM,data3)
    do_DTW(HMM,data4)


if __name__ == '__main__':
    train_hmm(sys.argv[1])
